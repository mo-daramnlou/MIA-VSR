import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch

# @ARCH_REGISTRY.register()
class WGEN67IVSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=24, num_blocks=4, out_channels=3, integrate_channels=16):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN67IVSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.fea_conv_ds = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=2)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre T convs
        self.ptconv2 = nn.Conv2d(mid_channels , mid_channels, kernel_size=1)
        self.ptconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels + 2 * integrate_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)
        self.tconv4 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1, groups=3)
        self.tconv5 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        self.btconv2 = nn.Conv2d(mid_channels , integrate_channels, kernel_size=1)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)

        # aT convs
        self.atconv2 = nn.Conv2d(mid_channels , integrate_channels, kernel_size=1)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        self.rconv= nn.Conv2d(mid_channels + 8 , mid_channels, kernel_size=1)

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
        lqs_batch = lqs.permute(0,3,1,2).view(10, 3, 180, 320).contiguous()

        image_skip = lqs_batch

        # Middle convolutions ds
        x_ds = self.relu(self.fea_conv_ds(lqs_batch))
        x_ds = F.interpolate(x_ds, scale_factor=2, mode='nearest')

        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))

        x = torch.cat([x,x_ds],dim=1)
        x = self.relu(self.rconv(x))

        # Middle convolutions
        feat_skip = x
        x = self.middle_convs(x)
        x = x + feat_skip

        # Pre T convs
        ptx = self.relu(self.ptconv2(x))
        ptx = self.relu(self.ptconv3(ptx))

        # bT convs
        btx = self.relu(self.btconv2(x))
        btx = self.relu(self.btconv3(btx))

        # aT convs
        atx = self.relu(self.atconv2(x))
        atx = self.relu(self.atconv3(atx))


        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_btx = torch.cat((btx[0:1], btx[:-1]), dim=0)

        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_atx = torch.cat((atx[1:], atx[10-1:10]), dim=0)
        
        x = torch.cat([shifted_btx,ptx,shifted_atx],dim=1)

        # T convs
        tx = self.relu(self.tconv1(x))
        tx = self.relu(self.tconv2(tx))
        tx = self.relu(self.tconv3(tx))
        tx = self.relu(self.tconv4(tx))
        tx = self.relu(self.tconv5(tx))


        # Pre-shuffle convolutions
        x = torch.cat((tx, image_skip), dim=1)

        x = self.relu(self.psconv(x))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x)
        
        output_batch = output_batch.view(1, 30, 720, 1280).permute(0,2,3,1).contiguous()
        
        return output_batch
       

if __name__ == '__main__':

    model = WGEN67IVSR(mid_channels=24, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/wgen51vsr31.tflite")