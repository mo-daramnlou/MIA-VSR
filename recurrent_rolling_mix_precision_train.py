import datetime
import logging
import math
import time
import torch
from os import path as osp
from torch.cuda.amp import GradScaler
import random
import threading

import archs
import data
import models
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, make_exp_dirs)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def create_val_dataloader(opt, logger):
    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
    return val_loaders

def load_resume_state(opt):
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
        return resume_state
    return None

def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    
    if opt['rank'] == 0: make_exp_dirs(opt)
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    train_set = build_dataset(opt['datasets']['train'])
    val_loaders = create_val_dataloader(opt, logger)
    total_iters = int(opt['train']['total_iter'])

    model = build_model(opt)
    resume_state = load_resume_state(opt)
    if resume_state:
        model.resume_training(resume_state)
        start_iter = resume_state['iter']
        logger.info(f"Resuming training from iter: {start_iter}.")
    else:
        start_iter = 0
    current_iter = start_iter

    msg_logger, scaler = MessageLogger(opt, current_iter, tb_logger), GradScaler()
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    
    logger.info(f'Start training from iter: {current_iter}')

    thread_result = {}
    def recreate_dataloader_thread_target(epoch_num):
        """Wrapper function to be run in a thread; creates a new DataLoader."""
        num_clips = opt['datasets']['train'].get('num_clips_per_epoch', 8)
        num_clips_to_load = min(num_clips, len(train_set.all_clip_names))
        
        sampled_clips = random.sample(train_set.all_clip_names, num_clips_to_load)
        train_set.load_clips_into_cache(sampled_clips)

        sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], opt['datasets']['train'].get('dataset_enlarge_ratio', 1))
        sampler.set_epoch(epoch_num)

        train_loader = build_dataloader(train_set, opt['datasets']['train'], num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=sampler)
        thread_result['dataloader'] = train_loader

    epoch = 0
    recreate_dataloader_thread_target(epoch)
    train_loader = thread_result['dataloader']

    while current_iter <= total_iters:
        for train_data in train_loader:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters: break
            
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warm_up_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(scaler, current_iter)
            iter_timer.record()

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
                # Delete the old dataloader and clear the cache to release memory
                del train_loader
                train_set.clear_cache()

                logger.info("--- Starting Validation and Asynchronous Data Pre-loading ---")
                epoch += 1
                loading_thread = threading.Thread(target=recreate_dataloader_thread_target, args=(epoch,))
                loading_thread.start()
                
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                
                logger.info("Validation complete. Waiting for data pre-loading to finish...")
                loading_thread.join()
                train_loader = thread_result['dataloader']
                logger.info("Data pre-loading finished. Resuming training.")
                break # Break from the inner 'for train_data' loop to start with the new loader
            
            data_timer.start()
            iter_timer.start()
        
        if current_iter > total_iters: break

    logger.info('End of training.')
    model.save(epoch=-1, current_iter=-1)
    if tb_logger: tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, '..'))
    train_pipeline(root_path)

