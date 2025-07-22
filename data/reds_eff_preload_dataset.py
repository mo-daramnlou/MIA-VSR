import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import h5py
import lmdb
import sys

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class REDSEffPreloadDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    MODIFIED FOR PRE-LOADING: This version pre-loads the entire dataset from
    LMDB into RAM during initialization to eliminate I/O bottlenecks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSEffPreloadDataset, self).__init__()
        print("init REDSRecurrentDistillationDataset")
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.is_lmdb = (self.opt['io_backend']['type'] == 'lmdb')
        
        logger = get_root_logger()

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]
            # self.keys = [v for v in self.keys]

        # --- PRE-LOADING LOGIC ---
        if not self.is_lmdb:
            raise ValueError('Pre-loading is only supported for LMDB backend.')

        logger.info('----------- Pre-loading REDS dataset into memory... -----------')
        self.preloaded_lq_data = {}
        self.preloaded_gt_data = {}
        
        # Get a set of unique clip names to iterate over
        unique_clips_to_load = sorted(list(set([k.split('/')[0] for k in self.keys])))
        logger.info(f'Found {len(unique_clips_to_load)} clips to load for pre-loading.')

        # Open LMDB environments
        env_lq = lmdb.open(str(self.lq_root), readonly=True, lock=False, readahead=False, meminit=False)
        env_gt = lmdb.open(str(self.gt_root), readonly=True, lock=False, readahead=False, meminit=False)

        self.each_clip_frame_number=60

        with env_lq.begin(write=False) as txn_lq, env_gt.begin(write=False) as txn_gt:
            for clip in unique_clips_to_load:
                for frame_idx in range(self.each_clip_frame_number):  # Each clip has 100 frames (0-99)
                    key = f'{clip}/{frame_idx:08d}'
                    
                    lq_img_bytes = txn_lq.get(key.encode('ascii'))
                    gt_img_bytes = txn_gt.get(key.encode('ascii'))
                    
                    if lq_img_bytes:
                        self.preloaded_lq_data[key] = imfrombytes(lq_img_bytes, float32=True)
                    if gt_img_bytes:
                        self.preloaded_gt_data[key] = imfrombytes(gt_img_bytes, float32=True)
        
        lq_size = sum(arr.nbytes for arr in self.preloaded_lq_data.values())
        gt_size = sum(arr.nbytes for arr in self.preloaded_gt_data.values())
        total_size_gb = (lq_size + gt_size) / (1024**3)
        logger.info(f'Successfully pre-loaded {len(self.preloaded_lq_data)} LQ and {len(self.preloaded_gt_data)} GT frames.')
        logger.info(f'Approximate memory usage for images: {total_size_gb:.2f} GB.')
        # --- END PRE-LOADING LOGIC ---

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        
        self.feature_hdf5_path = opt.get('feature_hdf5_path', None)
        self.kd_enabled = opt['kd_enabled']
        self.preloaded_features = {}
        if self.kd_enabled and self.feature_hdf5_path is not None:
            logger.info('Pre-loading knowledge distillation features...')
            with h5py.File(self.feature_hdf5_path, 'r') as hdf5_file:
                 for key in hdf5_file:
                    if key.split('/')[0] in unique_clips_to_load:
                        group = hdf5_file[key]
                        self.preloaded_features[key] = {
                            'backward_1': torch.from_numpy(group['backward_1'][()]).float(),
                            'forward_1': torch.from_numpy(group['forward_1'][()]).float(),
                            'backward_2': torch.from_numpy(group['backward_2'][()]).float(),
                            'forward_2': torch.from_numpy(group['forward_2'][()]).float(),
                        }
            logger.info(f'Successfully pre-loaded features for {len(self.preloaded_features)} frames.')
        
        logger.info('-------------------- Pre-loading complete --------------------')


    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > self.each_clip_frame_number - self.num_frame * interval:
            start_frame_idx = random.randint(0, self.each_clip_frame_number - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            frame_key = f'{clip_name}/{neighbor:08d}'
            # Use .copy() to prevent augmentations from modifying the master data in RAM
            img_lqs.append(self.preloaded_lq_data[frame_key].copy())
            img_gts.append(self.preloaded_gt_data[frame_key].copy())

        feature_maps_batch = {'backward_1': [], 'forward_1': [], 'backward_2': [], 'forward_2': []}
        if self.kd_enabled and self.preloaded_features:
            for neighbor in neighbor_list:
                frame_key = f'{clip_name}/{neighbor:08d}'
                if frame_key in self.preloaded_features:
                    features = self.preloaded_features[frame_key]
                    for module in feature_maps_batch.keys():
                        feature_maps_batch[module].append(features[module].clone())
            
            for module in feature_maps_batch:
                if feature_maps_batch[module]:
                    feature_maps_batch[module] = torch.stack(feature_maps_batch[module], dim=0)

        # randomly crop
        if self.kd_enabled:
            img_lqs.extend([f.unsqueeze(2).numpy() for f in feature_maps_batch['backward_1']])
            img_lqs.extend([f.unsqueeze(2).numpy() for f in feature_maps_batch['forward_1']])
            img_lqs.extend([f.unsqueeze(2).numpy() for f in feature_maps_batch['backward_2']])
            img_lqs.extend([f.unsqueeze(2).numpy() for f in feature_maps_batch['forward_2']])

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, f'{clip_name}/{neighbor_list[0]:08d}')

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        if self.kd_enabled:
            num_lq_frames = len(neighbor_list)
            img_lqs = torch.stack(img_results[:num_lq_frames], dim=0)
            feature_maps_batch['backward_1'] = torch.stack(img_results[num_lq_frames : 2*num_lq_frames], dim=0)
            feature_maps_batch['forward_1'] = torch.stack(img_results[2*num_lq_frames : 3*num_lq_frames], dim=0)
            feature_maps_batch['backward_2'] = torch.stack(img_results[3*num_lq_frames : 4*num_lq_frames], dim=0)
            feature_maps_batch['forward_2'] = torch.stack(img_results[4*num_lq_frames : 5*num_lq_frames], dim=0)
            img_gts = torch.stack(img_results[5*num_lq_frames:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:len(img_results) // 2], dim=0)
            img_gts = torch.stack(img_results[len(img_results) // 2:], dim=0)

        # reshape to match model input
        img_lqs = img_lqs.reshape(-1, img_lqs.shape[2], img_lqs.shape[3]).permute(1, 2, 0)
        img_gts = img_gts.reshape(-1, img_gts.shape[2], img_gts.shape[3]).permute(1, 2, 0)

        data = {'lq': img_lqs, 'gt': img_gts, 'key': key}
        if self.kd_enabled:
            data['feature_maps'] = feature_maps_batch

        return data

    def __len__(self):
        return len(self.keys)

    def normalize_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
