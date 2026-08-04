import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


import torch
import torch.nn as nn
import torch.nn.functional as F

class ECB_c(nn.Module):
    """
    Collapsible Edge-oriented Convolution Block (ECB)[cite: 12].

    This module implements the "efficient training methodology" from the
    SESR paper . Instead of running all four parallel 
    branches, this block calculates the arithmetically equivalent 
    single 3x3 kernel and bias in every forward pass.

    The backward pass will still update all the original expanded parameters.
    """
    def __init__(self, 
                 channels: int, 
                 expand_channels: int):
        """
        Initializes the expanded parameters for all four ECB branches.

        Args:
            channels (int): The number of input and output channels (C).
            expand_channels (int): The number of intermediate channels (D) 
                                   for Component II[cite: 182, 185].
        """
        super().__init__()
        self.channels = channels
        self.expand_channels = expand_channels

        # --- Component I: Normal 3x3 Conv --- [cite: 173, 178]
        # K_n, B_n
        self.conv_n = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # --- Component II: Expanding-and-Squeezing Conv --- [cite: 181, 183]
        # K_e, B_e (1x1 expand)
        self.conv_e = nn.Conv2d(channels, expand_channels, 1, padding=0, bias=True)
        # K_s, B_s (3x3 squeeze)
        self.conv_s = nn.Conv2d(expand_channels, channels, 3, padding=1, bias=True)

        # --- Component III: Scaled Sobel Filters --- [cite: 185-187]
        # K_x, B_x (1x1 conv for horizontal) [cite: 198]
        self.conv_x = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
        # K_y, B_y (1x1 conv for vertical) [cite: 198]
        self.conv_y = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
        
        # Learnable scaling factors and biases [cite: 199]
        self.S_Dx = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.B_Dx = nn.Parameter(torch.zeros(channels))
        self.S_Dy = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.B_Dy = nn.Parameter(torch.zeros(channels))

        # --- Component IV: Scaled Laplacian Filter --- [cite: 203]
        # K_l, B_l (1x1 conv) [cite: 210]
        self.conv_l = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
        # S_lap, B_lap (scaling and bias) [cite: 210]
        self.S_lap = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.B_lap = nn.Parameter(torch.zeros(channels))

        # --- Fixed Filters (Registered as non-learnable buffers) ---
        
        # Sobel D_x [cite: 191]
        D_x = torch.tensor([
            [+1.0, 0.0, -1.0], 
            [+2.0, 0.0, -2.0], 
            [+1.0, 0.0, -1.0]
        ]).view(1, 1, 3, 3)
        self.register_buffer('D_x', D_x)
        
        # Sobel D_y [cite: 192]
        D_y = torch.tensor([
            [+1.0, +2.0, +1.0], 
            [ 0.0,  0.0,  0.0], 
            [-1.0, -2.0, -1.0]
        ]).view(1, 1, 3, 3)
        self.register_buffer('D_y', D_y)

        # Laplacian D_lap [cite: 206]
        D_lap = torch.tensor([
            [0.0, +1.0, 0.0], 
            [+1.0, -4.0, +1.0], 
            [0.0, +1.0, 0.0]
        ]).view(1, 1, 3, 3)
        self.register_buffer('D_lap', D_lap)

    def _get_equivalent_kernel_bias(self):
        """
        Calculates the collapsed 3x3 kernel and bias from all branches.
        This logic is derived from Section 3.3 of the ECBSR paper .
        """
        # --- Component I kernel & bias ---
        K_n, B_n = self.conv_n.weight, self.conv_n.bias

        # --- Component II kernel & bias --- [cite: 217]
        K_e, B_e = self.conv_e.weight, self.conv_e.bias
        K_s, B_s = self.conv_s.weight, self.conv_s.bias
        
        # Merge weights: K_es = perm(K_e) * K_s
        # K_s shape (C, D, 3, 3) is input
        # perm(K_e) shape (C, D, 1, 1) is weight [cite: 219]
        # We use padding='same' for the 1x1 conv to align centers
        K_es_w = F.conv2d(
            input=K_s, 
            weight=K_e.permute(1, 0, 2, 3), 
            padding='same'
        ) 
        
        # Merge biases: B_es = K_s * rep(B_e) + B_s [cite: 217]
        B_e_rep = B_e.view(1, -1, 1, 1).repeat(1, 1, 3, 3) #[cite: 219]
        B_es_b = F.conv2d(B_e_rep, K_s).view(-1) + B_s

        # --- Component III kernel & bias (Sobel) --- [cite: 226, 228]
        K_x, B_x = self.conv_x.weight, self.conv_x.bias
        K_y, B_y = self.conv_y.weight, self.conv_y.bias
        C = self.channels
        
        # Create K_Dx (diagonal kernel from DW kernel) [cite: 223]
        K_Dx_w = torch.zeros(C, C, 3, 3, device=K_x.device)
        dw_kernel_x = (self.S_Dx * self.D_x) # Shape (C, 1, 3, 3) [cite: 200]
        K_Dx_w[range(C), range(C), :, :] = dw_kernel_x.squeeze(1)
        
        # Create K_Dy [cite: 223]
        K_Dy_w = torch.zeros(C, C, 3, 3, device=K_y.device)
        dw_kernel_y = (self.S_Dy * self.D_y)
        K_Dy_w[range(C), range(C), :, :] = dw_kernel_y.squeeze(1)

        # Merge weights: K_sob = perm(K_x)*K_Dx + perm(K_y)*K_Dy
        K_sob_x_w = F.conv2d(K_Dx_w, K_x.permute(1, 0, 2, 3), padding='same')
        K_sob_y_w = F.conv2d(K_Dy_w, K_y.permute(1, 0, 2, 3), padding='same')
        K_sob_w = K_sob_x_w + K_sob_y_w #[cite: 201]

        # Merge biases: B_sob = (K_Dx*rep(B_x) + B_Dx) + (K_Dy*rep(B_y) + B_Dy)
        B_x_rep = B_x.view(1, -1, 1, 1).repeat(1, 1, 3, 3)
        B_sob_x_b = F.conv2d(B_x_rep, K_Dx_w).view(-1) + self.B_Dx

        B_y_rep = B_y.view(1, -1, 1, 1).repeat(1, 1, 3, 3)
        B_sob_y_b = F.conv2d(B_y_rep, K_Dy_w).view(-1) + self.B_Dy
        B_sob_b = B_sob_x_b + B_sob_y_b

        # --- Component IV kernel & bias (Laplacian) --- [cite: 227, 229]
        K_l, B_l = self.conv_l.weight, self.conv_l.bias

        K_lap_w_diag = torch.zeros(C, C, 3, 3, device=K_l.device)
        dw_kernel_l = (self.S_lap * self.D_lap) #[cite: 207]
        K_lap_w_diag[range(C), range(C), :, :] = dw_kernel_l.squeeze(1)

        K_lap_w = F.conv2d(K_lap_w_diag, K_l.permute(1, 0, 2, 3), padding='same')

        B_l_rep = B_l.view(1, -1, 1, 1).repeat(1, 1, 3, 3)
        B_lap_b = F.conv2d(B_l_rep, K_lap_w_diag).view(-1) + self.B_lap

        # --- Final Summation --- [cite: 212, 226, 228]
        K_rep = K_n + K_es_w + K_sob_w + K_lap_w
        B_rep = B_n + B_es_b + B_sob_b + B_lap_b

        return K_rep, B_rep

    def forward(self, inputs):
        """
        Calculates the collapsed kernel/bias and performs a single 3x3 conv.
        """
        # Get the collapsed kernel and bias
        K_rep, B_rep = self._get_equivalent_kernel_bias()
        
        # Perform the final, efficient convolution [cite: 232]
        return F.conv2d(
            input=inputs, 
            weight=K_rep, 
            bias=B_rep, 
            stride=1, 
            padding="same"
        )
    

@ARCH_REGISTRY.register()
class WGEN57VSR(nn.Module):
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
        super(WGEN57VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.fea_conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.fea_conv_ds2 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, stride=2)
        self.fea_conv_ds2_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(ECB_c(mid_channels, 120))
            middle_layers.append(nn.ReLU(inplace=True))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre T convs
        self.ptconv2 = nn.Conv2d(mid_channels , mid_channels, kernel_size=1)
        self.ptconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        # self.ptconv4 = nn.Conv2d(integrate_channels, mid_channels, kernel_size=1)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels + 2 * integrate_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        # self.btconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv2 = nn.Conv2d(mid_channels , integrate_channels, kernel_size=1)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        # self.btconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        # self.atconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv2 = nn.Conv2d(mid_channels , integrate_channels, kernel_size=1)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        # self.atconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # self.middle_conv_ds = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)
        # middle_layers_ds1 = []
        # for _ in range(2):
        #     middle_layers_ds1.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        #     middle_layers_ds1.append(nn.ReLU(inplace=True))
        # self.middle_convs_ds1 = nn.Sequential(*middle_layers_ds1)

        # self.iconv = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1)
        self.rconv= nn.Conv2d(mid_channels + 8 , mid_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights similar to the Keras version."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("init: ",m)
                # glorot_normal initializer in Keras is Xavier normal in PyTorch
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # bias_initializer='zeros'
                    nn.init.zeros_(m.bias)
            else:
                print("pass: ",m)
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

        # Middle convolutions ds2
        x_ds2 = self.relu(self.fea_conv_ds2(lqs_batch))
        x_ds2 = self.relu(self.fea_conv_ds2_2(x_ds2))
        x_ds2 = F.interpolate(x_ds2, scale_factor=2, mode='nearest')

        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        x = self.relu(self.fea_conv_2(x))

        x = torch.cat([x,x_ds2],dim=1)
        x = self.relu(self.rconv(x))

        # Middle convolutions
        feat_skip = x
        x = self.middle_convs(x)
        x = x + feat_skip


        # Pre T convs
        ptx = self.relu(self.ptconv2(x))
        ptx = self.relu(self.ptconv3(ptx))
        # ptx = self.relu(self.ptconv4(ptx))

        # bT convs
        # btx = self.relu(self.btconv1(x))
        btx = self.relu(self.btconv2(x))
        btx = self.relu(self.btconv3(btx))
        # btx = self.relu(self.btconv4(btx))

        # aT convs
        # atx = self.relu(self.atconv1(x))
        atx = self.relu(self.atconv2(x))
        atx = self.relu(self.atconv3(atx))
        # atx = self.relu(self.atconv4(atx))


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
        
        # Clip the output to a valid image range
        # output_batch = torch.clamp(output_batch, max = 255.)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds
       

if __name__ == '__main__':

    model = WGEN57VSR(mid_channels=24, num_blocks=4)
    # model.eval()

    # Make test run
    # prediction = model(torch.randn(1, 180, 320, 9))
    # print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/wgen51vsr4.tflite")

    model.train() # Set to training mode to test collapsible block logic

    # Make test run
    # Example for training mode shape
    prediction = model(torch.randn(2, 64, 64, 30)) # N, H, W, T*C
    print(prediction.shape)

    model.eval() # Set to evaluation mode
    # Example for inference mode shape
    prediction = model(torch.randn(1, 10, 3, 180, 320)) # N, T, C, H, W
    print(prediction.shape)