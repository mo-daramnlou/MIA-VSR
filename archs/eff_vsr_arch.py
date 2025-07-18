import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from archs.custom_spynet_arch import SpyNet4Levels
from basicsr.utils.registry import ARCH_REGISTRY
from archs.mia_sliding_arch import SwinIRFM


@ARCH_REGISTRY.register()
class EFFVSR(nn.Module):
    """EFFVSR network structure.


    Paper:
        Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention

    """

    def __init__(self,
                in_channels=3,
                mid_channels=64,
                embed_dim=120,
                depths=(6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6),
                window_size=(3, 8, 8),
                num_frames=3,
                img_size = 64,
                patch_size=1,
                cpu_cache_length=100,
                is_low_res_input=True,
                use_mask=True,
                spynet_path=None):
        super().__init__()

        self.conv1 = nn.Conv2d(3, mid_channels, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(mid_channels, 3 * (4 ** 2), kernel_size=3, padding=1)     # 6 -> 48
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, lqs):

        """Forward function for MIAVSR.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape [32, 10, 3, 32, 32] 

        Returns:
            Tensor: Output HR sequence with shape  (32,10,3,128,128)
        """
        preds = []
        # print("lqs: ", lqs.shape)
        
        for i in range(lqs.shape[1]):
            lq = lqs[:, i, :, :, :]
            # print("lq: ", lq.shape)
            x1 = self.conv1(lq)
            x2 = self.conv2(x1)
            x3 = self.prelu(x2)
            x4 = self.conv3(x3)
            x5 = self.conv4(x4 + x1)
            x6 = self.pixel_shuffle(x5)
            preds.append(x6)
            
        preds = torch.stack(preds, dim=1)
        # print("preds: ", preds.shape) #[32, 8, 3, 128, 128] 1, 50, 3, 720, 1280
        # 32, 10, 3, 128, 128
        
        return preds, None, None


if __name__ == '__main__':
    #upscale = 4
    window_size = [3, 8, 8]
    img_size=128

    model = EFFVSR(
        mid_channels=6,
        embed_dim=32,
        depths=[4,4],
        num_heads=[4,4],
        window_size=window_size,
        num_frames = 3,
        img_size = img_size,
        patch_size = 1,
        cpu_cache_length = 100,
        is_low_res_input = True,
        use_mask=False,
        spynet_path='/home/mohammad/Documents/uni/deeplearning/FinalProject/MIA-VSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
        # spynet_path = '/content/MIA-VSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    )


    print(model)
    # macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, backend='aten',
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("flops",model.flops() / 1e9)

    x = torch.randn((1, 5, 3, img_size, img_size))
    x = model(x)
    print(x.shape)
