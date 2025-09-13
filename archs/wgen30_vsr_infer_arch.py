import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class WGEN30VSRInfer(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=24, num_blocks=6, out_channels=3, integrate_channels=16):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN30VSRInfer, self).__init__()
        self.scale = scale
        self.integrate_channels = integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers using standard Conv2d
        middle_layers1 = []
        for _ in range(int(num_blocks / 2)):
            middle_layers1.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            middle_layers1.append(nn.ReLU(inplace=True))
        self.middle_convs1 = nn.Sequential(*middle_layers1)

        middle_layers2 = []
        for _ in range(int(num_blocks / 2)):
            middle_layers2.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            middle_layers2.append(nn.ReLU(inplace=True))
        self.middle_convs2 = nn.Sequential(*middle_layers2)

        # Pre T convs
        self.ptconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.ptconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.ptconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.ptconv5 = nn.Conv2d(integrate_channels, mid_channels, kernel_size=1)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels + 2 * integrate_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        self.btconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.btconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.btconv5 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        self.atconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.atconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.atconv5 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights similar to the Keras version."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lqs):
        """
        Forward pass for inference.
        """
        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()

        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w).contiguous()

        image_skip = lqs_batch
        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip = x

        # Middle convolutions
        x = self.middle_convs1(x)
        x = self.middle_convs2(x + feat_skip)
        x = x + feat_skip

        # Pre T convs
        ptx = self.relu(self.ptconv2(x))
        ptx = self.relu(self.ptconv3(ptx))
        ptx = self.relu(self.ptconv4(ptx))
        ptx = self.relu(self.ptconv5(ptx))

        # bT convs
        btx = self.relu(self.btconv2(x))
        btx = self.relu(self.btconv3(btx))
        btx = self.relu(self.btconv4(btx))
        btx = self.relu(self.btconv5(btx))

        # aT convs
        atx = self.relu(self.atconv2(x))
        atx = self.relu(self.atconv3(atx))
        atx = self.relu(self.atconv4(atx))
        atx = self.relu(self.atconv5(atx))

        if self.training:
            btx = btx.view(n, t, self.integrate_channels, h, w).contiguous()
            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_btx = torch.cat((btx[:,0:1,:,:,:], btx[:,:-1,:,:,:]), dim=1)
            shifted_btx = shifted_btx.view(n*t, self.integrate_channels, h, w).contiguous()

            atx = atx.view(n, t, self.integrate_channels, h, w).contiguous()
            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_atx = torch.cat((atx[:,1:,:,:,:], atx[:,t-1:t,:,:,:]), dim=1)
            shifted_atx = shifted_atx.view(n*t, self.integrate_channels, h, w).contiguous()

        else:
            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_btx = torch.cat((btx[0:1], btx[:-1]), dim=0)

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_atx = torch.cat((atx[1:], atx[t-1:t]), dim=0)
        
        x = torch.cat([shifted_btx,ptx,shifted_atx],dim=1)

        # Assert proper concatenation
        # if self.training:
        #     btx = btx.view(n* t, self.integrate_channels, h, w).contiguous()
        #     atx = atx.view(n* t, self.integrate_channels, h, w).contiguous()
        #     for i,f in enumerate(x):
        #         if i%t == 0:
        #             assert torch.equal(f[0:self.integrate_channels], btx[i]), ('ass failed1')
        #         else:
        #             assert torch.equal(f[0:self.integrate_channels], btx[i-1]), ('ass failed2')

        #         if i%t == t-1:
        #             assert torch.equal(f[-self.integrate_channels:], atx[i]), ('ass failed3')
        #         else:
        #             assert torch.equal(f[-self.integrate_channels:], atx[i+1]), ('ass failed4')
        # else:
        #     for i,f in enumerate(x):
        #         if i == 0:
        #             assert torch.equal(f[0:self.integrate_channels], btx[i]), ('ass failed1')
        #         else:
        #             assert torch.equal(f[0:self.integrate_channels], btx[i-1]), ('ass failed2')

        #         if i == len(x)-1:
        #             assert torch.equal(f[-self.integrate_channels:], atx[i]), ('ass failed3')
        #         else:
        #             assert torch.equal(f[-self.integrate_channels:], atx[i+1]), ('ass failed4')

        # T convs
        tx = self.relu(self.tconv1(x))
        tx = self.relu(self.tconv2(tx))
        tx = self.relu(self.tconv3(tx))

        # Pre-shuffle convolutions
        x = torch.cat((tx, image_skip), dim=1)
        x = self.relu(self.psconv(x))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x)

        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)

        return preds
