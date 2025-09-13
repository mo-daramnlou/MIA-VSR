import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


@ARCH_REGISTRY.register()
class WGEN7INFERVSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3, integrate_channels=28):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN7INFERVSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre T convs
        self.ptconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups= mid_channels)
        self.ptconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels + 2 * integrate_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        # self.btconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1, groups= integrate_channels)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        # self.atconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1, groups= integrate_channels)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # self.iconv = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1)

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
        #  print("lqs: ", lqs.shape) 1, 180, 320, 30

        # is_train_mode = len(lqs.shape) == 4
        # if is_train_mode:
        #     n, h, w, tc = lqs.shape
        #     lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        # n, t, c, h, w = lqs.shape
        # lqs_batch = lqs.view(n * t, c, h, w).contiguous() #320, 3, 64, 64
        
        lqs_batch = lqs.permute(0,3,1,2).view(10, 3, 180, 320).contiguous()

        image_skip = lqs_batch
        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip=x
        
        # Middle convolutions
        x = self.middle_convs(x)
        x = x + feat_skip

        # Pre T convs
        ptx = self.relu(self.ptconv2(x))
        ptx = self.relu(self.ptconv3(ptx))

        # bT convs
        # btx = self.relu(self.btconv1(x))
        btx = self.relu(self.btconv2(x))
        btx = self.relu(self.btconv3(btx))

        # aT convs
        # atx = self.relu(self.atconv1(x))
        atx = self.relu(self.atconv2(x))
        atx = self.relu(self.atconv3(atx))


        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_btx = torch.cat((btx[0:1], btx[:-1]), dim=0)

        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_atx = torch.cat((atx[1:], atx[10-1:10]), dim=0)
        
        x = torch.cat([shifted_btx,ptx,shifted_atx],dim=1)

       
        for i,f in enumerate(x):
            if i == 0:
                assert torch.equal(f[0:self.integrate_channels], btx[i]), ('ass failed1')
            else:
                assert torch.equal(f[0:self.integrate_channels], btx[i-1]), ('ass failed2')

            if i == len(x)-1:
                assert torch.equal(f[-self.integrate_channels:], atx[i]), ('ass failed3')
            else:
                assert torch.equal(f[-self.integrate_channels:], atx[i+1]), ('ass failed4')


        # T convs
        tx = self.relu(self.tconv1(x))
        tx = self.relu(self.tconv2(tx))
        tx = self.relu(self.tconv3(tx))


        # Pre-shuffle convolutions
        x = torch.cat((tx, image_skip), dim=1)

        x = self.relu(self.psconv(x))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x) #10, 3, 180, 320
        
        # Clip the output to a valid image range
        # output_batch = torch.clamp(output_batch, max = 255.)


        # --- Output Shape Handling ---
        # _, c_out, h_out, w_out = output_batch.shape
        # preds = output_batch.view(n, t, c_out, h_out, w_out)

        # if is_train_mode:
        #     preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # # print("preds: ", preds.shape) #32, 256, 256, 30


        output_batch = output_batch.view(1, 30, 720, 1280).permute(0,2,3,1).contiguous()
        
        return output_batch


if __name__ == '__main__':

    model = WGEN7INFERVSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")