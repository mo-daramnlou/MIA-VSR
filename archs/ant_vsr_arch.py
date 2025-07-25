import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3-conv1x1':
            # Define the intermediate channel numbers.
            self.mid_planes1 = int(inp_planes * depth_multiplier)
            self.mid_planes2 = int(inp_planes * depth_multiplier)

            # First 1x1 convolution
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes1, kernel_size=1, padding=0)
            self.k0 = nn.Parameter(conv0.weight)
            self.b0 = nn.Parameter(conv0.bias)

            # 3x3 convolution
            conv1 = torch.nn.Conv2d(self.mid_planes1, self.mid_planes2, kernel_size=3)
            self.k1 = nn.Parameter(conv1.weight)
            self.b1 = nn.Parameter(conv1.bias)

            # Second 1x1 convolution
            conv2 = torch.nn.Conv2d(self.mid_planes2, self.out_planes, kernel_size=1, padding=0)
            self.k2 = nn.Parameter(conv2.weight)
            self.b2 = nn.Parameter(conv2.bias)

        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if not self.training:
             raise ValueError("Forward pass should not be called in eval mode. Use rep_params instead.")
        if self.type == 'conv1x1-conv3x3-conv1x1':
            # First 1x1 conv
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)

            # 3x3 conv with explicit padding to match re-parameterization logic
            y0_padded = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0_padded[:, :, 0:1, :] = b0_pad
            y0_padded[:, :, -1:, :] = b0_pad
            y0_padded[:, :, :, 0:1] = b0_pad
            y0_padded[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0_padded, weight=self.k1, bias=self.b1, stride=1)

            # Second 1x1 conv
            y2 = F.conv2d(input=y1, weight=self.k2, bias=self.b2, stride=1)
            return y2


    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3-conv1x1':
            # Step 1: Fuse 1x1 conv (k0, b0) and 3x3 conv (k1, b1)
            K_01 = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            B_01_padded_input = torch.ones(1, self.mid_planes1, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            B_01 = F.conv2d(input=B_01_padded_input, weight=self.k1).view(-1,) + self.b1

            # Step 2: Fuse the result with the second 1x1 conv (k2, b2)
            # We need to treat the 1x1 conv as a matrix multiplication
            k2_mat = self.k2.squeeze(3).squeeze(2) # Shape: (out_planes, mid_planes2)
            # Reshape the 3x3 kernel into a matrix
            K_01_mat = K_01.view(self.mid_planes2, -1)
            RK_mat = k2_mat @ K_01_mat
            RK = RK_mat.view(self.out_planes, self.inp_planes, 3, 3)

            # Fuse bias: B_final = k2 * B_01 + b2
            RB = k2_mat @ B_01 + self.b2
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
class ANTVSR(nn.Module):
    """AntSR network structure for quantized image super-resolution.
    
    As described in the Mobile AI 2025 Challenge Report:
    "Quantized Image Super-Resolution on Mobile NPUs"
    
    Key features:
    - Reparameterized convolution blocks for efficient inference
    - Skip connection to mitigate quantization loss
    - Clamping and PixelShuffle for final upscaling
    - Designed for real-time performance on mobile NPUs
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, mid_channels=32, num_blocks=4, upscale=4):
        super().__init__()
        self.upscale = upscale
        
        # Initial convolution
        self.conv_first = nn.Conv2d(num_in_ch, mid_channels, kernel_size=3, padding=1)
        
        # Reparameterizable convolution blocks
        self.body = nn.ModuleList([
            RepConvBlock(inp_planes = mid_channels, out_planes= mid_channels, act_type="relu", with_idt=False) for _ in range(num_blocks)
        ])
        
        # Final convolution (output channels = upscale² * 3)
        self.conv_final = nn.Conv2d(
            mid_channels, 
            num_out_ch * (upscale ** 2), 
            kernel_size=3, 
            padding=1
        )
        
        # Pixel shuffle for upscaling
        self.upsample = nn.PixelShuffle(upscale)
        
        # ReLU activation (used consistently in the architecture)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lqs):
        # --- Input Shape Handling ---

        # print("lqs: ", lqs.shape) # 32, 64, 64, 30 --  1, 3, 720, 1280
        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w) #320, 3, 64, 64
        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64


        # Initial convolution
        feat = self.relu(self.conv_first(lqs_batch))
        skip_connection = feat
        
        # Process through RepConv blocks
        backbone_feat = feat
        for block in self.body:
            backbone_feat = block(backbone_feat)
        
        # Skip connection
        feat = backbone_feat + skip_connection
        
        # Final convolution, clamp, and upscale
        feat = self.relu(self.conv_final(feat))
        # if not self.training:  SINCE I HAVEN'T USED CLIPPING FOR OTHER MODELS, FOR ACURATE COMPARISSON I DON'T USE IT HERE AS WELL
        #     feat = torch.clamp(feat, max=255)  # Prevent overflow as described in the paper
        output_batch = self.upsample(feat)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds, None, None


if __name__ == '__main__':
    # Test the AntSR model with the winning configuration from the challenge
    model = ANTVSR(
        mid_channels=8,  # Hidden channels (C in the paper)
        num_blocks=4,   # Number of RepConv blocks
        upscale=4     # 3x upscaling
    )
    model.train()  # Start in training mode

    print("AntSR Model Architecture:")
    print(model)
    
    # Test with a validation-style input tensor
    test_input = torch.randn(1, 180, 320, 30)  # HD input (360p)
    print(f"\nInput shape: {test_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape} (should be 1080p: 1080x1920)")
    
    # Test reparameterization
    # model.reparameterize()
    print("\nAfter reparameterization:")
    model.eval()
    print(f"Is training mode: {model.training}")
    
    # Test inference after reparameterization
    with torch.no_grad():
        rep_output = model(test_input)

    
    # Verify outputs match
    print(f"Output match before/after reparameterization: "
          f"{torch.allclose(output, rep_output, atol=1e-5)}")
    difference = torch.sum(torch.abs(output - rep_output))
    print(f"Sum of absolute difference between outputs: {difference.item()}")