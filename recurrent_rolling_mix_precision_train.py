import datetime
import logging
import math
import torch.profiler
import time
import torch
from os import path as osp
from torch.cuda.amp import GradScaler
import random
import threading

import archs  # noqa F401
import data  # noqa F401
import models  # noqa F401
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # ... (code is unchanged)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # --- MODIFIED TO USE NEW SUBSECTION LOGIC FOR EPOCH ESTIMATION ---
    train_loader, val_loaders = None, []
    train_set = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # --- FIX: Estimate iterations per epoch using cache_section_frames ---
            if 'num_clips_per_epoch' in dataset_opt: 
                num_clips = dataset_opt['num_clips_per_epoch']
                # Use the new parameter for a correct estimation
                frames_per_section = dataset_opt.get('cache_section_frames', 30) 
                num_frame_sequence = dataset_opt.get('num_frame', 21)
                sequences_per_epoch = num_clips * (frames_per_section - num_frame_sequence + 1)
                num_iter_per_epoch = math.ceil(
                    sequences_per_epoch * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            else: # Fallback for standard datasets
                num_iter_per_epoch = math.ceil(
                    len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))

            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) if num_iter_per_epoch > 0 else 0
            
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)} (Note: 0 for rolling cache until first load)'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tEst. iterations per epoch: {num_iter_per_epoch}'
                        f'\n\tEst. total epochs: {total_epochs}; total iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters, train_set


def load_resume_state(opt):
    # ... (code is unchanged)
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
    # ... (code is unchanged)
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    resume_state = load_resume_state(opt)
    if resume_state is None: make_exp_dirs(opt)
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)
    _, _, val_loaders, total_epochs, total_iters, train_set = create_train_val_dataloader(opt, logger)
    # train_set = build_dataset(opt['datasets']['train'])
    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch, current_iter = resume_state['epoch'], resume_state['iter']
    else:
        start_epoch, current_iter = 0, 0
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    scaler = GradScaler()
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    thread_result = {}
    def recreate_loader_and_prefetcher(epoch_num):
        num_clips = opt['datasets']['train'].get('num_clips_per_epoch', 8)
        num_clips_to_load = min(num_clips, len(train_set.all_clip_names))
        sampled_clips = random.sample(train_set.all_clip_names, num_clips_to_load)
        train_set.load_clips_into_cache(sampled_clips)
        sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], opt['datasets']['train'].get('dataset_enlarge_ratio', 1))
        sampler.set_epoch(epoch_num)
        train_loader = build_dataloader(train_set, opt['datasets']['train'], num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=sampler)
        thread_result['dataloader'] = train_loader
    recreate_loader_and_prefetcher(start_epoch)
    train_loader = thread_result['dataloader']
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, total_epochs + 1):
        for train_data in train_loader:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters: break
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(scaler, current_iter)
            iter_timer.record()
            if current_iter == 1: msg_logger.reset_start_time()
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'lrs': model.get_current_learning_rate(),
                            'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()}
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                logger.info("--- Clearing old cache before validation ---")
                del train_loader
                train_set.clear_cache()
                logger.info("--- Starting Validation and Asynchronous Data Pre-loading ---")
                loading_thread = threading.Thread(target=recreate_loader_and_prefetcher, args=(epoch + 1,))
                loading_thread.start()
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                logger.info("Validation complete. Waiting for data pre-loading to finish...")
                loading_thread.join()
                train_loader = thread_result['dataloader']
                logger.info("Data pre-loading finished. Resuming training.")
                break 
            data_timer.start()
            iter_timer.start()
        if current_iter > total_iters: break
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
    root_path = osp.abspath(osp.join(__file__, osp.pardir, '..'))
    train_pipeline(root_path)

