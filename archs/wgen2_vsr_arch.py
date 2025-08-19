import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


@ARCH_REGISTRY.register()
class WGEN2VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN2VSR, self).__init__()
        self.scale = scale
        self.mid_channels=mid_channels
        self.mid_channels=mid_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # T convs
        self.tconv1 = nn.Conv2d(3 * mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # # bT convs
        # self.btconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        # self.btconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.btconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # # aT convs
        # self.atconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        # self.atconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.atconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # self.iconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights similar to the Keras version."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # glorot_normal initializer in Keras is Xavier normal in PyTorch
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # bias_initializer='zeros'
                    nn.init.zeros_(m.bias)

    def forward(self, lqs):
        """
        Forward pass.
        Note: PyTorch uses (N, C, H, W) channel order, while the TensorFlow
        model used (N, H, W, C). The model is adapted for the PyTorch convention.
        """
        #  print("lqs: ", lqs.shape) # 32, 64, 64, 30 --  1, 3, 720, 1280

        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w).contiguous() #320, 3, 64, 64

        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64

        # t, c, h, w = lqs.shape
        image_skip = lqs_batch
        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip=x
        
        # Middle convolutions
        x = self.middle_convs(x)
        x = x + feat_skip


        if self.training:
            x_view = x.view(n, t, self.mid_channels, h, w).contiguous()

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_bx = torch.cat((x_view[:,0:1,:,:,:], x_view[:,:-1,:,:,:]), dim=1)
            shifted_bx = shifted_bx.view(n*t, self.mid_channels, h, w).contiguous()

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_ax = torch.cat((x_view[:,1:,:,:,:], x_view[:,t-1:t,:,:,:]), dim=1)
            shifted_ax = shifted_ax.view(n*t, self.mid_channels, h, w).contiguous()

        else:
            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_bx = torch.cat((x[0:1], x[:-1]), dim=0)

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_ax = torch.cat((x[1:], x[t-1:t]), dim=0)
        

        # Pre-shuffle convolutions
        conx = torch.cat((shifted_bx ,x, shifted_ax), dim=1)

        # print(conx.shape)
        # Assert proper concatenation
        
        
        if self.training:
            x_view = x.view(n* t, self.mid_channels, h, w).contiguous()
            for i,f in enumerate(conx):
                if i%t == 0:
                    assert torch.equal(f[0:self.mid_channels], x_view[i]), ('ass failed1')
                else:
                    assert torch.equal(f[0:self.mid_channels], x_view[i-1]), ('ass failed2')

                if i%t == t-1:
                    assert torch.equal(f[-self.mid_channels:], x_view[i]), ('ass failed3')
                else:
                    assert torch.equal(f[-self.mid_channels:], x_view[i+1]), ('ass failed4')
        else:
            for i,f in enumerate(conx):
                if i == 0:
                    assert torch.equal(f[0:self.mid_channels], x[i]), ('ass failed1')
                else:
                    assert torch.equal(f[0:self.mid_channels], x[i-1]), ('ass failed2')

                if i == len(x)-1:
                    assert torch.equal(f[-self.mid_channels:], x[i]), ('ass failed3')
                else:
                    assert torch.equal(f[-self.mid_channels:], x[i+1]), ('ass failed4')



        # T convs
        x = self.relu(self.tconv1(conx))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
        x = self.relu(self.psconv(torch.cat([x,image_skip],dim=1)))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x)
        
        # Clip the output to a valid image range
        # output_batch = torch.clamp(output_batch, max = 255.)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds, None, None


if __name__ == '__main__':

    model = WGEN2VSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(10, 3, 180, 320),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")