import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import h5py

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
import os.path as osp

@DATASET_REGISTRY.register()
class REDSRecurrentDistillationDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

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
        super(REDSRecurrentDistillationDataset, self).__init__()
        print("init REDSRecurrentDistillationDataset")
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

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

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        
        self.feature_hdf5_path = opt.get('feature_hdf5_path', None)
        self.feature_hdf5 = None
        self.kd_enabled = opt['kd_enabled']

    def __getitem__(self, index):
        # print("getitem REDSRecurrentDistillationDataset")
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if self.kd_enabled and self.feature_hdf5_path is not None:
            self.feature_hdf5 = h5py.File(self.feature_hdf5_path, 'r')

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        
        feature_maps_batch = {'backward_1': [], 'forward_1': [], 'backward_2': [], 'forward_2': []} # 4modules; each num_frame, 180, 320
        if self.kd_enabled and self.feature_hdf5 is not None:
            # save_folder = f'/home/mohammad/Documents/uni/deep learning/FinalProject/inference_data/kd_train/'
            for neighbor in neighbor_list:
                group = self.feature_hdf5[f"{clip_name}/{neighbor:08d}"]
                for module in feature_maps_batch.keys():
                    fmap = group[module][()]  # numpy array
                    feature_maps_batch[module].append(torch.from_numpy(fmap).float())
                    # feat = tensor2img(self.normalize_tensor(torch.from_numpy(fmap).float()), rgb2bgr=True, min_max=(0,1))
                    # # feat = anchor_feat[idx].squeeze()
                    # # print("feat: ",feat)
                    # s_folder = osp.join(save_folder, f"{clip_name}_{neighbor:08d}_{module}" + '.png')
                    # imwrite(feat, s_folder)
            # Stack to tensors: (num_frames, H, W)
            for module in feature_maps_batch:
                feature_maps_batch[module] = torch.stack(feature_maps_batch[module], dim=0)

        # randomly crop
        if self.kd_enabled:
            img_lqs.extend(feature_maps_batch['backward_1'].unsqueeze(3).numpy())
            img_lqs.extend(feature_maps_batch['forward_1'].unsqueeze(3).numpy())
            img_lqs.extend(feature_maps_batch['backward_2'].unsqueeze(3).numpy())
            img_lqs.extend(feature_maps_batch['forward_2'].unsqueeze(3).numpy())

        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        if self.kd_enabled:
            img_lqs = torch.stack(img_results[:len(img_results) // 6], dim=0)
            feature_maps_batch['backward_1'] = torch.stack(img_results[1*(len(img_results) // 6): 2*(len(img_results) // 6)], dim=0)
            feature_maps_batch['forward_1'] = torch.stack(img_results[2*(len(img_results) // 6): 3*(len(img_results) // 6)], dim=0)
            feature_maps_batch['backward_2'] = torch.stack(img_results[3*(len(img_results) // 6): 4*(len(img_results) // 6)], dim=0)
            feature_maps_batch['forward_2'] = torch.stack(img_results[4*(len(img_results) // 6): 5*(len(img_results) // 6)], dim=0)
            img_gts = torch.stack(img_results[5*(len(img_results) // 6):], dim=0)
        else:
            img_lqs = torch.stack(img_results[:len(img_results) // 2], dim=0)
            img_gts = torch.stack(img_results[len(img_results) // 2:], dim=0)
        

        # save_folder = f'/home/mohammad/Documents/uni/deep learning/FinalProject/inference_data/kd_train2/'
        # for i in range(20,24):
        #     print("shape: ",img_results[i].shape)
        #     feat = tensor2img(self.normalize_tensor(torch.Tensor(img_results[i]).squeeze()), rgb2bgr=True, min_max=(0,1))
        #     # feat = anchor_feat[idx].squeeze()
        #     # print("feat: ",feat)
        #     s_folder = osp.join(save_folder, f"{np.random.randint(100)}" + '.png')
        #     imwrite(feat, s_folder)
        

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str

        data = {
            'lq': img_lqs,
            'gt': img_gts,
            'key': key
        }
        if self.kd_enabled:
            data['feature_maps'] = feature_maps_batch if self.feature_hdf5 is not None else None

        return data

    def __len__(self):
        return len(self.keys)
    
    def normalize_tensor(self,tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor