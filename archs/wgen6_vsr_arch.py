import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchsummary import summary
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


# This is the low-level implementation of the re-parameterizable convolutions.
# It remains unchanged from the original file.
class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            # Initialize weights with Xavier normal and biases to zero
            nn.init.xavier_normal_(self.conv0.weight)
            if self.conv0.bias is not None:
                nn.init.zeros_(self.conv0.bias)
            # self.k0 = conv0.weight
            # self.b0 = conv0.bias

            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            # Initialize weights with Xavier normal and biases to zero
            nn.init.xavier_normal_(self.conv1.weight)
            if self.conv1.bias is not None:
                nn.init.zeros_(self.conv1.bias)
            # self.k1 = conv1.weight
            # self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            # Initialize weights with Xavier normal and biases to zero
            nn.init.xavier_normal_(self.conv0.weight)
            if self.conv0.bias is not None:
                nn.init.zeros_(self.conv0.bias)
            # self.k0 = conv0.weight
            # self.b0 = conv0.bias

            # init scale & bias
            self.scale = nn.Parameter(torch.empty(self.out_planes, 1, 1, 1))
            nn.init.xavier_normal_(self.scale)
            self.bias = nn.Parameter(torch.empty(self.out_planes))
            nn.init.zeros_(self.bias)
            
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            # Initialize weights with Xavier normal and biases to zero
            nn.init.xavier_normal_(self.conv0.weight)
            if self.conv0.bias is not None:
                nn.init.zeros_(self.conv0.bias)
            # self.k0 = conv0.weight
            # self.b0 = conv0.bias

            # init scale & bias
            self.scale = nn.Parameter(torch.empty(self.out_planes, 1, 1, 1))
            nn.init.xavier_normal_(self.scale)
            self.bias = nn.Parameter(torch.empty(self.out_planes))
            nn.init.zeros_(self.bias)

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            # Initialize weights with Xavier normal and biases to zero
            nn.init.xavier_normal_(self.conv0.weight)
            if self.conv0.bias is not None:
                nn.init.zeros_(self.conv0.bias)
            # self.k0 = conv0.weight
            # self.b0 = conv0.bias

            # init scale & bias
            self.scale = nn.Parameter(torch.empty(self.out_planes, 1, 1, 1))
            nn.init.xavier_normal_(self.scale)
            self.bias = nn.Parameter(torch.empty(self.out_planes))
            nn.init.zeros_(self.bias)

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            k0 = self.conv0.weight
            b0 = self.conv0.bias
            k1 = self.conv1.weight
            b1 = self.conv1.bias

            y0 = F.conv2d(input=x, weight=k0, bias=b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=k1, bias=b1, stride=1)
        else:
            k0 = self.conv0.weight
            b0 = self.conv0.bias

            y0 = F.conv2d(input=x, weight=k0, bias=b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        k0 = self.conv0.weight
        b0 = self.conv0.bias        
        
        device = k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            k1 = self.conv1.weight
            b1 = self.conv1.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB

# The Edge-oriented Convolution Block (ECB) for training.
# It remains unchanged from the original file.
class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


@ARCH_REGISTRY.register()
class WGEN6VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3, integrate_channels=16):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN6VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            # middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            # middle_layers.append(nn.ReLU(inplace=True))
            middle_layers.append(ECB(mid_channels, mid_channels, depth_multiplier=2.0, act_type='relu', with_idt=False))
        self.middle_convs = nn.Sequential(*middle_layers)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        self.btconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv2 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        self.atconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv2 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3 + (integrate_channels * 2), out_channels * (scale**2), kernel_size=1)

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
                # print("init:",m)
                # glorot_normal initializer in Keras is Xavier normal in PyTorch
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # bias_initializer='zeros'
                    nn.init.zeros_(m.bias)
            # else:
            #     print("pass:",m)

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
        
        # T convs
        tx = self.relu(self.tconv1(x))
        tx = self.relu(self.tconv2(tx))
        tx = self.relu(self.tconv3(tx))

        # bT convs
        btx = self.relu(self.btconv1(x))
        btx = self.relu(self.btconv2(btx))
        btx = self.relu(self.btconv3(btx))

        # bT convs
        atx = self.relu(self.atconv1(x))
        atx = self.relu(self.atconv2(atx))
        atx = self.relu(self.atconv3(atx))


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
        

        # Pre-shuffle convolutions
        x = torch.cat((shifted_btx ,tx, image_skip, shifted_atx), dim=1)


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
        
        return preds, None, None


def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() )


if __name__ == '__main__':

    model = WGEN6VSR(mid_channels=28, num_blocks=4)
    # model.eval()

    # Make test run
    # prediction = model(torch.randn(1, 180, 320, 30))
    # print(prediction.shape)

    # Converting model to TFLite

    # sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # summary(model, (1, 3, 180, 320))
    num_params = count_trainable_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")


    # model.train()  # Start in training mode

    # print("AntSR Model Architecture:")
    # print(model)
    
    # # Test with a validation-style input tensor
    # test_input = torch.randn(1, 180, 320, 30)  # HD input (360p)
    # print(f"\nInput shape: {test_input.shape}")
    
    # # Test forward pass
    # with torch.no_grad():
    #     output = model(test_input)
    
    # print(f"Output shape: {output.shape} (should be 1080p: 1080x1920)")
    
    # # Test reparameterization
    # # model.reparameterize()
    # print("\nAfter reparameterization:")
    # model.eval()
    # print(f"Is training mode: {model.training}")
    
    # # Test inference after reparameterization
    # with torch.no_grad():
    #     rep_output = model(test_input)

    
    # # Verify outputs match
    # print(f"Output match before/after reparameterization: "
    #       f"{torch.allclose(output, rep_output, atol=1e-5)}")
    # difference = torch.sum(torch.abs(output - rep_output))
    # print(f"Sum of absolute difference between outputs: {difference.item()}")
