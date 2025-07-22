import datetime
import logging
import math
import torch.profiler
import time
import torch
from os import path as osp
from torch.cuda.amp import GradScaler

import archs  # noqa F401
import data  # noqa F401
import models  # noqa F401
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

# +++ DALI Import (New) +++
from data.reds_eff_dali_dataset import DALIDataLoader
# +++ End DALI Import +++


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # +++ DALI Integration (Modified) +++
            use_dali = dataset_opt.get('use_dali', False)
            if use_dali:
                logger.info('Using NVIDIA DALI for data loading.')
                train_loader = DALIDataLoader(opt)
                train_sampler = None # DALI handles its own sampling
                dataset_enlarge_ratio = 1 # DALI does not use enlarge ratio
                num_train_samples = len(train_loader.dali_iterator)
            else:
                logger.info('Using PyTorch dataloader for data loading.')
                dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
                train_set = build_dataset(dataset_opt)
                num_train_samples = len(train_set)
                train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
                train_loader = build_dataloader(
                    train_set,
                    dataset_opt,
                    num_gpu=opt['num_gpu'],
                    dist=opt['dist'],
                    sampler=train_sampler,
                    seed=opt['manual_seed'])
            # +++ End DALI Integration +++

            num_iter_per_epoch = math.ceil(
                num_train_samples * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {num_train_samples}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # +++ DALI Integration (Modified) +++
    # Bypass prefetcher when using DALI
    use_dali = opt['datasets']['train'].get('use_dali', False)
    if not use_dali:
        prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
        if prefetch_mode is None or prefetch_mode == 'cpu':
            prefetcher = CPUPrefetcher(train_loader)
        elif prefetch_mode == 'cuda':
            prefetcher = CUDAPrefetcher(train_loader, opt)
            logger.info(f'Use {prefetch_mode} prefetch dataloader')
            if opt['datasets']['train'].get('pin_memory') is not True:
                raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        else:
            raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.')
    # +++ End DALI Integration +++

    scaler = GradScaler()
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # +++ DALI Integration (Modified) +++
        # Choose the correct iterator
        if use_dali:
            data_iterator = iter(train_loader)
        else:
            prefetcher.reset()
            data_iterator = iter(prefetcher)
        # +++ End DALI Integration +++

        for train_data in data_iterator:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break
            
            # --- Handle data format from DALI ---
            if use_dali:
                # DALI returns a list of dictionaries
                train_data = train_data[0]
                # DALI reshapes sequences, need to adjust for model input
                # From [N, T, C, H, W] -> [N, H, W, T*C]
                n, t, c, h, w = train_data['lq'].shape
                train_data['lq'] = train_data['lq'].permute(0, 3, 4, 1, 2).reshape(n, h, w, t*c)
                train_data['gt'] = train_data['gt'].permute(0, 3, 4, 1, 2).reshape(n, h, w, t*c)
            # --- End data format handling ---

            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(scaler, current_iter)
            iter_timer.record()

            if current_iter == 1 and not resume_state:
                msg_logger.reset_start_time()

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
