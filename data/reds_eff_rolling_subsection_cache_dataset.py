import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import lmdb
import gc

from basicsr.utils import get_root_logger, imfrombytes
from basicsr.utils.registry import DATASET_REGISTRY

# --- GPU-accelerated augmentation functions (unchanged) ---
def gpu_paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    lq_patch_size = gt_patch_size // scale
    t, c, h_lq, w_lq = img_lqs.size()
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)
    img_lqs = img_lqs[:, :, top:top + lq_patch_size, left:left + lq_patch_size]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = img_gts[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]
    return img_gts, img_lqs

def gpu_augment(tensors, hflip=False, vflip=False, rot90=False):
    if hflip: tensors = torch.flip(tensors, dims=[3])
    if vflip: tensors = torch.flip(tensors, dims=[2])
    if rot90: tensors = tensors.transpose(2, 3)
    return tensors

@DATASET_REGISTRY.register()
class REDSEffRollingSubsectionCacheDataset(data.Dataset):
    """
    Rolling cache dataset that loads random SUBSECTIONS of clips.
    This increases data diversity when memory is a constraint.
    """
    def __init__(self, opt):
        super(REDSEffRollingSubsectionCacheDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- NEW: Parameters for subsection sampling ---
        self.total_frames_per_clip = opt.get('total_frames_per_clip', 100)
        self.cache_section_frames = opt.get('cache_section_frames', 30)

        logger = get_root_logger()
        all_keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                # Ensure we only consider clips that have the specified number of frames
                if int(frame_num) == self.total_frames_per_clip:
                    all_keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])
        
        self.all_clip_names = sorted(list(set([k.split('/')[0] for k in all_keys])))

        val_partition_map = {'REDS4': ['000', '011', '015', '020'], 'official': [f'{v:03d}' for v in range(240, 270)]}
        val_partition = val_partition_map.get(opt['val_partition'], [])
        
        # if not opt['test_mode']:
        #     self.all_clip_names = [v for v in self.all_clip_names if v not in val_partition]

        self.vram_cache_lq, self.vram_cache_gt = {}, {}
        self.valid_sequences = []
        
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)

        # --- OPTIMIZATION: Open LMDB environments once and keep them open ---
        self.env_lq = lmdb.open(str(self.lq_root), readonly=True, lock=False, readahead=False, meminit=False)
        self.env_gt = lmdb.open(str(self.gt_root), readonly=True, lock=False, readahead=False, meminit=False)

        logger.info(f'Dataset initialized with {len(self.all_clip_names)} total clips. Ready to load subsections on-demand.')

    def clear_cache(self):
        # ... (code is unchanged)
        logger = get_root_logger()
        logger.info("Clearing VRAM cache and running garbage collection...")
        self.vram_cache_lq.clear()
        self.vram_cache_gt.clear()
        self.valid_sequences = []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def load_clips_into_cache(self, clips_to_load):
        self.clear_cache()
        logger = get_root_logger()
        logger.info(f"----------- Loading {len(clips_to_load)} new random subsections into VRAM... -----------")
        
        cached_sections = {} # Store the random start frame for each loaded clip

        with self.env_lq.begin(write=False) as txn_lq, self.env_gt.begin(write=False) as txn_gt:
            cursor_lq = txn_lq.cursor()
            cursor_gt = txn_gt.cursor()

            for clip in clips_to_load:
                section_start = random.randint(0, self.total_frames_per_clip - self.cache_section_frames)
                cached_sections[clip] = section_start

                start_key = f'{clip}/{section_start:08d}'.encode('ascii')
                if not cursor_lq.set_key(start_key) or not cursor_gt.set_key(start_key):
                    logger.warning(f"Could not find start key {start_key.decode('ascii')} for clip {clip}. Skipping.")
                    continue

                for i in range(self.cache_section_frames):
                    key_bytes, lq_img_bytes = cursor_lq.item()
                    _, gt_img_bytes = cursor_gt.item()
                    
                    key_str = key_bytes.decode('ascii')
                    lq_img, gt_img = imfrombytes(lq_img_bytes, float32=True), imfrombytes(gt_img_bytes, float32=True)
                    self.vram_cache_lq[key_str] = torch.from_numpy(lq_img).permute(2, 0, 1).to(self.device)
                    self.vram_cache_gt[key_str] = torch.from_numpy(gt_img).permute(2, 0, 1).to(self.device)
                    
                    cursor_lq.next()
                    cursor_gt.next()

        # --- NEW: Build valid sequences based on the loaded subsections ---
        interval = max(self.interval_list)
        for clip_name, section_start in cached_sections.items():
            # The last possible start frame for a sequence within the subsection
            last_possible_start = (section_start + self.cache_section_frames) - (self.num_frame * interval)
            for start_frame_idx in range(section_start, last_possible_start + 1):
                self.valid_sequences.append((clip_name, start_frame_idx))

        logger.info(f"----------- VRAM cache loaded. {len(self.valid_sequences)} valid sequences available. -----------")

    def __getitem__(self, index):
        # ... (code is unchanged)
        clip_name, start_frame_idx = self.valid_sequences[index]
        scale, gt_size = self.opt['scale'], self.opt['gt_size']
        interval = random.choice(self.interval_list)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        if self.random_reverse and random.random() < 0.5: neighbor_list.reverse()

        img_lqs_list, img_gts_list = [], []
        for neighbor in neighbor_list:
            frame_key = f'{clip_name}/{neighbor:08d}'
            img_lqs_list.append(self.vram_cache_lq[frame_key].clone().detach())
            img_gts_list.append(self.vram_cache_gt[frame_key].clone().detach())

        img_lqs, img_gts = torch.stack(img_lqs_list, dim=0), torch.stack(img_gts_list, dim=0)
        img_gts, img_lqs = gpu_paired_random_crop(img_gts, img_lqs, gt_size, scale)
        hflip, vflip, rot90 = (self.opt['use_hflip'] and random.random() < 0.5,
                               self.opt['use_rot'] and random.random() < 0.5,
                               self.opt['use_rot'] and random.random() < 0.5)
        img_lqs, img_gts = gpu_augment(img_lqs, hflip, vflip, rot90), gpu_augment(img_gts, hflip, vflip, rot90)
        img_lqs = img_lqs.reshape(-1, img_lqs.shape[2], img_lqs.shape[3]).permute(1, 2, 0)
        img_gts = img_gts.reshape(-1, img_gts.shape[2], img_gts.shape[3]).permute(1, 2, 0)
        return {'lq': img_lqs, 'gt': img_gts, 'key': f'{clip_name}/{start_frame_idx:08d}'}

    def __len__(self):
        return len(self.valid_sequences)

