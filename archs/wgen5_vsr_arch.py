import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3-conv1x1':
            self.mid_planes1 = int(inp_planes * depth_multiplier)
            self.mid_planes2 = int(inp_planes * depth_multiplier)

            # Store the Conv2d layers as class members
            self.conv0 = nn.Conv2d(self.inp_planes, self.mid_planes1, kernel_size=1, padding=0)
            self.conv1 = nn.Conv2d(self.mid_planes1, self.mid_planes2, kernel_size=3)
            self.conv2 = nn.Conv2d(self.mid_planes2, self.out_planes, kernel_size=1, padding=0)

        else:
            raise ValueError('the type of seqconv is not supported!')
    
    def forward(self, x):
        if not self.training:
             raise ValueError("Forward pass should not be called in eval mode. Use rep_params instead.")
        if self.type == 'conv1x1-conv3x3-conv1x1':
            # First 1x1 conv
            k0, b0 = self.conv0.weight, self.conv0.bias
            k1, b1 = self.conv1.weight, self.conv1.bias
            k2, b2 = self.conv2.weight, self.conv2.bias
            y0 = F.conv2d(input=x, weight=k0, bias=b0, stride=1)

            # 3x3 conv with explicit padding to match re-parameterization logic
            y0_padded = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = b0.view(1, -1, 1, 1)
            y0_padded[:, :, 0:1, :] = b0_pad
            y0_padded[:, :, -1:, :] = b0_pad
            y0_padded[:, :, :, 0:1] = b0_pad
            y0_padded[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0_padded, weight=k1, bias=b1, stride=1)

            # Second 1x1 conv
            y2 = F.conv2d(input=y1, weight=k2, bias=b2, stride=1)
            return y2

    def rep_params(self):
        # Update rep_params to use the weights from the stored modules
        k0, b0 = self.conv0.weight, self.conv0.bias
        k1, b1 = self.conv1.weight, self.conv1.bias
        k2, b2 = self.conv2.weight, self.conv2.bias
        
        # ... (rest of the rep_params logic using k0, b0, etc.)
        # The logic below this comment remains the same, just ensure variables are sourced correctly.
        device = k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3-conv1x1':
            K_01 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            B_01_padded_input = torch.ones(1, self.mid_planes1, 3, 3, device=device) * b0.view(1, -1, 1, 1)
            B_01 = F.conv2d(input=B_01_padded_input, weight=k1).view(-1,) + b1

            k2_mat = k2.squeeze(3).squeeze(2)
            K_01_mat = K_01.view(self.mid_planes2, -1)
            RK_mat = k2_mat @ K_01_mat
            RK = RK_mat.view(self.out_planes, self.inp_planes, 3, 3)

            RB = k2_mat @ B_01 + b2
            return RK, RB
        

# The Edge-oriented Convolution Block (RepConv) for training.
# This is modified to have two branches: (1x1->3x3->1x1) and (1x1).
class RepConvBlock(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=1, act_type='prelu', with_idt = False):
        super(RepConvBlock, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        # Branch 1: 1x1 conv -> 3x3 conv -> 1x1 conv
        self.branch1 = SeqConv3x3('conv1x1-conv3x3-conv1x1', self.inp_planes, self.out_planes, self.depth_multiplier)
        
        # Branch 2: 1x1 conv
        self.branch2 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)

        # self.rep_params_cache = None


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
        # print("self.training: ",self.training)
        if self.training:
            # Calculate outputs of both branches
            y1 = self.branch1(x)
            y2 = self.branch2(x)
            
            # Sum the outputs
            y = y1 + y2
            
            # Add identity connection if specified
            if self.with_idt:
                y += x
        else:
            # In eval mode, use the fused kernel and bias
            # if self.rep_params_cache is None:
            #     print("using cached params")
            #     self.rep_params_cache = self.rep_params()
            # RK, RB = self.rep_params_cache
            RK, RB = self.rep_params()
            # The padding is 1 for a 3x3 kernel
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)

        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        # Fuse Branch 1 (1x1 -> 3x3 -> 1x1)
        K1, B1 = self.branch1.rep_params()
        
        # Fuse Branch 2 (1x1)
        K2, B2 = self.branch2.weight, self.branch2.bias
        # Pad 1x1 kernel to 3x3. The 1x1 kernel is at the center.
        K2_padded = F.pad(K2, [1,1,1,1])

        # Total fused kernel and bias from the two branches
        RK = K1 + K2_padded
        RB = B1 + B2

        # Fuse the identity branch if it exists
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            # Identity kernel is a 3x3 kernel with 1 at the center
            K_idt = torch.zeros(self.out_planes, self.inp_planes, 3, 3, device=device)
            # This is correct since with_idt requires inp_planes == out_planes
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        
        return RK, RB


@ARCH_REGISTRY.register()
class WGEN5VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3, integrate_channels=8):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN5VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            # middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            # middle_layers.append(nn.ReLU(inplace=True))
            middle_layers.append(RepConvBlock(inp_planes = mid_channels, out_planes= mid_channels, act_type="relu", with_idt=False))
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
                print("init: ",m)

                # glorot_normal initializer in Keras is Xavier normal in PyTorch
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # bias_initializer='zeros'
                    nn.init.zeros_(m.bias)
            # elif isinstance(m, SeqConv3x3):
            #     # print("init: ",m)
            #     m.initialize()
            else:
                print("pass:", m)
            

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
        
        
        if self.training:
            btx = btx.view(n* t, self.integrate_channels, h, w).contiguous()
            atx = atx.view(n* t, self.integrate_channels, h, w).contiguous()
            for i,f in enumerate(x):
                if i%t == 0:
                    assert torch.equal(f[0:self.integrate_channels], btx[i]), ('ass failed1')
                else:
                    assert torch.equal(f[0:self.integrate_channels], btx[i-1]), ('ass failed2')

                if i%t == t-1:
                    assert torch.equal(f[-self.integrate_channels:], atx[i]), ('ass failed3')
                else:
                    assert torch.equal(f[-self.integrate_channels:], atx[i+1]), ('ass failed4')
        else:
            for i,f in enumerate(x):
                if i == 0:
                    assert torch.equal(f[0:self.integrate_channels], btx[i]), ('ass failed1')
                else:
                    assert torch.equal(f[0:self.integrate_channels], btx[i-1]), ('ass failed2')

                if i == len(x)-1:
                    assert torch.equal(f[-self.integrate_channels:], atx[i]), ('ass failed3')
                else:
                    assert torch.equal(f[-self.integrate_channels:], atx[i+1]), ('ass failed4')



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


if __name__ == '__main__':

    model = WGEN5VSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    # sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")


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