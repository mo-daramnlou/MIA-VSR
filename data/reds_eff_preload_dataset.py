import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import h5py
import lmdb
import sys
import gc # Import garbage collector

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

def gpu_paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop on GPU tensors, replicating basicsr logic."""
    lq_patch_size = gt_patch_size // scale
    t, c, h_lq, w_lq = img_lqs.size()
    
    # Randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # Crop lq patch
    img_lqs = img_lqs[:, :, top:top + lq_patch_size, left:left + lq_patch_size]

    # Crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = img_gts[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]
    
    return img_gts, img_lqs


def gpu_augment(tensors, hflip=False, vflip=False, rot90=False):
    """Augmentation on GPU tensors, replicating basicsr logic."""
    if hflip:
        tensors = torch.flip(tensors, dims=[3])
    if vflip:
        tensors = torch.flip(tensors, dims=[2])
    if rot90:
        tensors = tensors.transpose(2, 3)

    return tensors


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
            # self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]
            self.keys = [v for v in self.keys if v.split('/')[0]]

        # --- PRE-LOADING LOGIC ---
        if not self.is_lmdb:
            raise ValueError('Pre-loading is only supported for LMDB backend.')

        # --- PRE-LOADING LOGIC (CPU) ---
        logger.info('----------- Pre-loading REDS dataset into CPU RAM... -----------')
        self.preloaded_lq_data_cpu = {}
        self.preloaded_gt_data_cpu = {}
        
        unique_clips_to_load = sorted(list(set([k.split('/')[0] for k in self.keys])))
        logger.info(f'Found {len(unique_clips_to_load)} clips to load for pre-loading.')

        env_lq = lmdb.open(str(self.lq_root), readonly=True, lock=False, readahead=False, meminit=False)
        env_gt = lmdb.open(str(self.gt_root), readonly=True, lock=False, readahead=False, meminit=False)

        self.each_clip_frame_number = 25 
        with env_lq.begin(write=False) as txn_lq, env_gt.begin(write=False) as txn_gt:
            for clip in unique_clips_to_load:
                for frame_idx in range(self.each_clip_frame_number):
                    key = f'{clip}/{frame_idx:08d}'
                    lq_img_bytes = txn_lq.get(key.encode('ascii'))
                    gt_img_bytes = txn_gt.get(key.encode('ascii'))
                    
                    if lq_img_bytes and gt_img_bytes:
                        lq_img = imfrombytes(lq_img_bytes, float32=True)
                        gt_img = imfrombytes(gt_img_bytes, float32=True)
                        self.preloaded_lq_data_cpu[key] = torch.from_numpy(lq_img).permute(2, 0, 1)
                        self.preloaded_gt_data_cpu[key] = torch.from_numpy(gt_img).permute(2, 0, 1)
        
        logger.info('----------- Pre-loading to CPU RAM complete. -----------')

        # --- VRAM CACHING LOGIC ---
        self.vram_cache_lq = {}
        self.vram_cache_gt = {}
        vram_cache_gb = self.opt.get('vram_cache_gb', 0)
        if vram_cache_gb > 0:
            logger.info(f'----------- Caching portion of dataset to VRAM ({vram_cache_gb} GB)... -----------')
            
            total_cached_bytes = 0
            target_cache_bytes = vram_cache_gb * (1024**3)

            for key in self.preloaded_lq_data_cpu:
                if total_cached_bytes >= target_cache_bytes:
                    break
                
                lq_tensor = self.preloaded_lq_data_cpu[key]
                gt_tensor = self.preloaded_gt_data_cpu[key]

                self.vram_cache_lq[key] = lq_tensor.to(self.device)
                self.vram_cache_gt[key] = gt_tensor.to(self.device)
                total_cached_bytes += lq_tensor.element_size() * lq_tensor.nelement()
                total_cached_bytes += gt_tensor.element_size() * gt_tensor.nelement()
            
            logger.info(f'Cached {len(self.vram_cache_lq)} frames to VRAM. Actual size: {total_cached_bytes / (1024**3):.2f} GB.')

        # --- MODIFICATION: Conditionally clear CPU RAM ---
        if self.opt.get('clear_cpu_ram', False):
            logger.info('Clearing CPU RAM cache as requested...')
            del self.preloaded_lq_data_cpu
            del self.preloaded_gt_data_cpu
            gc.collect()
            logger.info('CPU RAM cache cleared.')
        # --- END MODIFICATION ---
        
        logger.info('-------------------- Dataset initialization complete --------------------')

        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        interval = random.choice(self.interval_list)
        start_frame_idx = int(frame_name)
        if start_frame_idx > self.each_clip_frame_number - self.num_frame * interval:
            start_frame_idx = random.randint(0, self.each_clip_frame_number - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        img_lqs_list = []
        img_gts_list = []
        for neighbor in neighbor_list:
            frame_key = f'{clip_name}/{neighbor:08d}'
            
            # Check VRAM cache first
            if frame_key in self.vram_cache_lq:
                img_lqs_list.append(self.vram_cache_lq[frame_key].clone())
                img_gts_list.append(self.vram_cache_gt[frame_key].clone())
            else:
                # This fallback will only work if `clear_cpu_ram` is false.
                # If true, it will raise a KeyError, indicating an insufficient VRAM cache.
                img_lqs_list.append(self.preloaded_lq_data_cpu[frame_key].to(self.device))
                img_gts_list.append(self.preloaded_gt_data_cpu[frame_key].to(self.device))

        # Stack frames into a sequence tensor on the GPU
        img_lqs = torch.stack(img_lqs_list, dim=0)
        img_gts = torch.stack(img_gts_list, dim=0)

        # Perform augmentations on the GPU
        img_gts, img_lqs = gpu_paired_random_crop(img_gts, img_lqs, gt_size, scale)
        
        hflip = self.opt['use_hflip'] and random.random() < 0.5
        vflip = self.opt['use_rot'] and random.random() < 0.5
        rot90 = self.opt['use_rot'] and random.random() < 0.5

        # Apply the same augmentations to both tensors separately
        img_lqs = gpu_augment(img_lqs, hflip, vflip, rot90)
        img_gts = gpu_augment(img_gts, hflip, vflip, rot90)

        # Reshape to match model input
        img_lqs = img_lqs.reshape(-1, img_lqs.shape[2], img_lqs.shape[3]).permute(1, 2, 0)
        img_gts = img_gts.reshape(-1, img_gts.shape[2], img_gts.shape[3]).permute(1, 2, 0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
