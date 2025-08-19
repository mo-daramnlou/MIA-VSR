import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# # Mock registry for standalone execution
# def register_mock(cls):
#     return cls

# ARCH_REGISTRY = type('obj', (object,), {'register': register_mock})

@ARCH_REGISTRY.register()
class GEN3VSR(nn.Module):
    """
    Original GEN3VSR model with the feature skip connection.
    """
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3):
        super(GEN3VSR, self).__init__()
        self.scale = scale
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)
        self.tconv1 = nn.Conv2d(mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lqs_batch):
        image_skip = lqs_batch
        feat_skip = self.relu(self.fea_conv(lqs_batch))
        x = self.middle_convs(feat_skip)
        x = x + feat_skip
        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
        x = torch.cat((x, image_skip), dim=1)
        x = self.relu(self.psconv(x))
        out = self.pixel_shuffle(x)
        output_batch = torch.clamp(out, min=0., max=255.)
        return output_batch

@ARCH_REGISTRY.register()
class GEN3VSR_ET(nn.Module):
    """
    Equivalent Transformation (ET) version of GEN3VSR.
    The feat_skip 'add' operation is removed and its logic is folded into tconv1.
    """
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3):
        super(GEN3VSR_ET, self).__init__()
        self.scale = scale
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)
        
        # MODIFIED: tconv1 now takes 2 * mid_channels as input
        self.tconv1 = nn.Conv2d(mid_channels * 2, out_channels * (scale**2), kernel_size=1)
        
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lqs_batch):
        image_skip = lqs_batch
        feat_skip = self.relu(self.fea_conv(lqs_batch))
        x = self.middle_convs(feat_skip)
        
        # MODIFIED: Concatenate instead of adding
        x = torch.cat([x, feat_skip], dim=1)
        
        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
        x = torch.cat((x, image_skip), dim=1)
        x = self.relu(self.psconv(x))
        out = self.pixel_shuffle(x)
        # Clamp is removed for the plain network; it's handled in verification
        return out
