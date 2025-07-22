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
class BIVSR(nn.Module):
    """BIVSR network structure.


    Paper:
        Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention

    """

    def __init__(self):
        super().__init__()


    def forward(self, lqs):

        """Forward function for MIAVSR.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with 4,180,320,30
                shape 

        Returns:
            Tensor: Output HR sequence with shape 
        """
        preds = []
        # print("lqs: ", lqs.shape) #32, 64, 64, 30

        mode ="train"
        if len(lqs.shape) == 4:
            mode = "train"
        else:
            mode = "val"


        if mode=="train": 
            n, h, w, tc = lqs.shape
            lqs = lqs.reshape(n, h, w, tc//3, 3)
            lqs = lqs.permute(0, 3, 4, 1, 2)
        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64

        for i in range(lqs.shape[1]):
            lq = lqs[:, i, :, :, :]
            # print("lq: ", lq.shape)
            x = torch.nn.functional.interpolate(lq, scale_factor=4, mode='bicubic')
            preds.append(x)
            
        preds = torch.stack(preds, dim=1)
        # print("preds: ", preds.shape) #[32, 8, 3, 128, 128] 1, 50, 3, 720, 1280
        # 32, 10, 3, 128, 128

        if mode == "train":
            n, t, c, h, w = preds.shape
            preds = preds.permute(0, 3, 4, 1, 2)
            preds = preds.reshape(n, h, w, t*c)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds, None, None


if __name__ == '__main__':
    model = BIVSR()


    print(model)
    # macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, backend='aten',
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("flops",model.flops() / 1e9)

    x = torch.randn((1, 5, 3, img_size, img_size))
    x = model(x)
    print(x.shape)
