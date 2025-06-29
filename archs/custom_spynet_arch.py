import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class BasicModule(nn.Module):
    """Basic module of SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


@ARCH_REGISTRY.register()
class SpyNet4Levels(nn.Module):
    """A custom SpyNet with 4 levels (3 downsampling steps) for 32x32 inputs.

    Args:
        load_path (str): Path to the pre-trained weights of a standard 5-level SpyNet.
            The weights for the first 4 levels will be loaded.
    """

    def __init__(self, load_path):
        super(SpyNet4Levels, self).__init__()
        self.level1 = BasicModule()
        self.level2 = BasicModule()
        self.level3 = BasicModule()
        self.level4 = BasicModule()

        # Load weights from the original 5-level SpyNet
        if load_path:
            state_dict = torch.load(load_path)
            self.load_state_dict(state_dict, strict=False)

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(3):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        # Level 4 (coarsest)
        flow = ref[0].new_zeros([ref[0].size(0), 2, int(ref[0].size(2)), int(ref[0].size(3))])
        flow = self.level4(torch.cat([ref[0], F.interpolate(input=supp[0], size=(int(ref[0].size(2)), int(ref[0].size(3))), mode='bilinear', align_corners=False), flow], 1))

        # Level 3
        flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
        flow = self.level3(torch.cat([ref[1], F.interpolate(input=supp[1], size=(int(ref[1].size(2)), int(ref[1].size(3))), mode='bilinear', align_corners=False), flow], 1))

        # Level 2
        flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
        flow = self.level2(torch.cat([ref[2], F.interpolate(input=supp[2], size=(int(ref[2].size(2)), int(ref[2].size(3))), mode='bilinear', align_corners=False), flow], 1))

        # Level 1 (finest)
        flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
        flow = self.level1(torch.cat([ref[3], F.interpolate(input=supp[3], size=(int(ref[3].size(2)), int(ref[3].size(3))), mode='bilinear', align_corners=False), flow], 1))

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = w // 32 * 32
        h_floor = h // 32 * 32

        if w_floor != w or h_floor != h:
            ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
            supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = self.process(ref, supp)

        if w_floor != w or h_floor != h:
            flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)

        return flow


# This is needed to make sure the new arch is registered
import importlib
from os import path as osp

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in sorted(os.listdir(arch_folder)) if v.endswith('_arch.py')]
_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]