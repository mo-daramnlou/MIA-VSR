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
# import ai_edge_torch


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
        self.mid_channels= mid_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, mid_channels, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(mid_channels, 3 * (4 ** 2), kernel_size=3, padding=1)     # 6 -> 48
        self.pixel_shuffle = nn.PixelShuffle(4)

    # def forward(self, lqs):

    #     """Forward function for MIAVSR.

    #     Args:
    #         lqs (tensor): Input low quality (LQ) sequence with 4,180,320,30
    #             shape 

    #     Returns:
    #         Tensor: Output HR sequence with shape 
    #     """
    #     preds = []
    #     # print("lqs: ", lqs.shape)
        
    #     for i in range(lqs.shape[1]):
    #         lq = lqs[:, i, :, :, :]
    #         # print("lq: ", lq.shape)
    #         x1 = self.conv1(lq)
    #         x2 = self.conv2(x1)
    #         x3 = self.prelu(x2)
    #         x4 = self.conv3(x3)
    #         x5 = self.conv4(x4 + x1)
    #         x6 = self.pixel_shuffle(x5)
    #         preds.append(x6)
            
    #     preds = torch.stack(preds, dim=1)
    #     # print("preds: ", preds.shape) #[32, 8, 3, 128, 128] 1, 50, 3, 720, 1280
    #     # 32, 10, 3, 128, 128
        
    #     return preds, None, None
    

    def forward(self, lqs):
        """
        Forward function for RCBSR, adapted for efficient video processing.
        It processes all frames in a batch simultaneously.
        """
        # --- Input Shape Handling ---

        # print("lqs: ", lqs.shape) # 32, 64, 64, 30 --  1, 3, 720, 1280

        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w) #320, 3, 64, 64

        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64

        x = self.relu(self.conv1(lqs_batch))
        skip = x
        x = self.conv2(x)
        x = self.prelu(x) #320, 6, 64, 64
        

        # # --- Start of Modified Frame Concatenation Logic ---
        # # 1. Reshape to separate batch and time dimensions.
        # # Shape changes from (320, 6, 64, 64) to (32, 10, 6, 64, 64).
        # x_reshaped = x.view(n, t, self.mid_channels, h, w)

        # # 2. Create the t-1, t, and t+1 sequences along the time dimension (dim=1).
        # # This operation is now performed independently for each of the 32 batches.

        # # For each batch, prepend its first frame to frames 0 through t-2.
        # x_prev = torch.cat([x_reshaped[:, 0:1, ...], x_reshaped[:, :-1, ...]], dim=1)

        # # The 't' sequence is just the original reshaped tensor.
        # x_curr = x_reshaped

        # # For each batch, append its last frame to frames 1 through t-1.
        # x_next = torch.cat([x_reshaped[:, 1:, ...], x_reshaped[:, -1:, ...]], dim=1)

        # # 3. Concatenate the three tensors along the channel dimension (dim=2).
        # # Shape becomes: (32, 10, 18, 64, 64).
        # x_concat = torch.cat((x_prev, x_curr, x_next), dim=2)

        # # 4. Reshape back to the flattened format for the next conv layer.
        # # Shape becomes: (320, 18, 64, 64).
        # x = x_concat.view(n * t, self.mid_channels * 3, h, w)
        # # --- End of Modified Logic ---


        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x + skip))
        output_batch = self.pixel_shuffle(x) #320, 3, 256, 256


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30

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

    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/effvsr30.tflite")




    # print(model)
    # # macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, backend='aten',
    # #                                        print_per_layer_stat=True, verbose=True)
    # # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("flops",model.flops() / 1e9)

    # x = torch.randn((1, 5, 3, img_size, img_size))
    # x = model(x)
    # print(x.shape)
