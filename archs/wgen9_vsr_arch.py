import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


# EXPANDED Linear block
class LinearBlock_e(nn.Module):
    """
    Expanded linear block (PyTorch version).
    Input --> 3x3 Conv to expand number of channels to 'feature_size' --> 1x1 Conv to project channels into 'out_filters'.

    At inference time, this can be analytically collapsed into a single,
    small 3x3 Conv layer. This implementation is straightforward but less
    efficient for training compared to the collapsed version.
    """
    def __init__(self,
                 in_filters: int,
                 num_inner_layers: int,
                 kernel_size: int,
                 padding: str,
                 out_filters: int,
                 feature_size: int,
                 quant_W: bool, # Note: quant_W is kept for signature consistency.
                 mode: str):
        super().__init__()

        # In PyTorch, 'same' padding for an odd kernel with stride 1 is calculated as kernel_size // 2
        def get_padding(kernel_size_):
            if padding == 'same':
                return kernel_size_ // 2
            elif padding == 'valid':
                return 0
            else:
                raise ValueError(f"Unsupported padding type: {padding}")

        layers = []
        # The first layer expands the channels.
        layers.append(nn.Conv2d(in_channels=in_filters,
                                out_channels=feature_size,
                                kernel_size=kernel_size,
                                padding=get_padding(kernel_size)))
        
        # Additional inner layers, if any.
        for _ in range(num_inner_layers - 1):
             layers.append(nn.Conv2d(in_channels=feature_size,
                                     out_channels=feature_size,
                                     kernel_size=kernel_size,
                                     padding=get_padding(kernel_size)))

        # The final layer projects the channels down to the output size.
        layers.append(nn.Conv2d(in_channels=feature_size,
                                out_channels=out_filters,
                                kernel_size=1,
                                padding=0)) # 1x1 conv with 'valid' padding
        
        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)


# COLLAPSED Linear block
class LinearBlock_c(nn.Module):
    """
    This is a simulated linear block in the train path (PyTorch version).
    The idea is to collapse the linear block at each training step to speed
    up the forward pass. The backward pass still updates all the expanded weights.

    After training is completed, the weight generation ops are replaced by
    a constant tensor at inference time.
    """
    def __init__(self,
                 in_filters: int,
                 num_inner_layers: int,
                 kernel_size: int,
                 padding: str,
                 out_filters: int,
                 feature_size: int,
                 quant_W: bool, # Note: quant_W is kept for signature consistency.
                 mode: str):
        super().__init__()

        # --- Parameters ---
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.feature_size = feature_size
        self.kx = kernel_size
        self.ky = kernel_size
        
        # Note: The quantization logic from the original TF implementation is omitted
        # as it relies on specific TF functions. The quant_W flag is kept for API consistency.
        self.quant_W = quant_W

        # If num_inner_layers > 1, then use another conv1x1 at the beginning
        onebyone = True if num_inner_layers > 1 else False

        # --- Learnable Convs for Collapsing ---
        # These layers will operate on the small 'delta' tensor to generate
        # the collapsed weights, not on the main feature maps.
        convs = []
        if onebyone:
            convs.append(nn.Conv2d(in_channels=self.in_filters, out_channels=self.feature_size, kernel_size=1, padding='valid'))
        
        convs.append(nn.Conv2d(in_channels=self.feature_size if onebyone else self.in_filters,
                               out_channels=self.feature_size,
                               kernel_size=(self.ky, self.kx),
                               padding='valid'))
        
        convs.append(nn.Conv2d(in_channels=self.feature_size, out_channels=self.out_filters, kernel_size=1, padding='valid'))
        
        self.collapse = nn.Sequential(*convs)
        self.collapsed_weights = None

        # --- Delta Tensor for Weight Generation ---
        # This tensor acts as an input to the 'collapse' sequence to generate the final weights.
        # It's registered as a buffer, so it's part of the module's state but not a trainable parameter.
        delta = torch.eye(self.in_filters)
        delta = delta.view(self.in_filters, self.in_filters, 1, 1)
        # Pad the delta tensor to prepare it for the 'valid' convolutions in the collapse sequence.
        delta_padded = F.pad(delta, [self.kx - 1, self.kx - 1, self.ky - 1, self.ky - 1])
        self.register_buffer('delta', delta_padded)

        # --- Residual Tensor ---
        # This tensor represents the identity connection (skip connection) that will be
        # arithmetically added to the generated weights.
        residual = torch.zeros(self.out_filters, self.in_filters, self.ky, self.kx)
        if self.in_filters == self.out_filters:
            mid_kx = self.kx // 2
            mid_ky = self.ky // 2
            for i in range(self.out_filters):
                residual[i, i, mid_ky, mid_kx] = 1.0
        self.register_buffer('residual', residual)

    def forward(self, inputs):
        # The 'self.training' flag is a built-in PyTorch attribute from nn.Module.
        # It's True during training (.train() mode) and False during evaluation (.eval() mode).
        # if self.training or self.collapsed_weights is None:
            # --- Online Linear Collapse ---
            # 1. Pass the delta tensor through the conv sequence to get the un-transposed weights.
        wt_tensor = self.collapse(self.delta)

        # 2. Reverse the spatial dimensions of the generated weights.
        wt_tensor = torch.flip(wt_tensor, dims=[2, 3])

        # 3. Transpose the dimensions to match PyTorch's Conv2d weight format.
        wt_tensor = wt_tensor.permute(1, 0, 2, 3)

        # 4. Add the residual (identity connection) directly to the weights.
        wt_tensor = wt_tensor + self.residual

        # 5. During inference, cache the collapsed weights
        if not self.training:
            self.collapsed_weights = nn.Parameter(wt_tensor, requires_grad=False)
        # else:
        #     # Use the pre-computed and cached weights during inference.
        #     wt_tensor = self.collapsed_weights

        # --- Final Convolution ---
        return F.conv2d(inputs, wt_tensor, stride=1, padding="same")



@ARCH_REGISTRY.register()
class WGEN9VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3, integrate_channels=28, expand_size=256):
        """
        PyTorch implementation of the base7 TensorFlow model, modified to use
        collapsible linear blocks in the middle layers.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            mid_channels (int): Number of feature channels for the main path.
            num_blocks (int): Number of middle collapsible blocks.
            out_channels (int): Number of channels in the output image.
            integrate_channels (int): Channels for temporal integration.
            feature_size (int): The internal expansion dimension for collapsible blocks.
        """
        super(WGEN9VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Replaced standard Conv2d layers with the efficient LinearBlock_c.
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(LinearBlock_c(
                in_filters=mid_channels,
                num_inner_layers=1,  # As per the SESR paper for 3x3 blocks
                kernel_size=3,
                padding='same',
                out_filters=mid_channels,
                feature_size=expand_size, # Internal expansion dimension
                quant_W=False,
                mode='train' # The block handles train/eval switching internally
            ))
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

        # --- BUG FIX: Unify temporal shifting logic for train and eval ---
        # The previous 'else' block handled evaluation incorrectly. This unified
        # logic works for any batch size n >= 1 in both modes.
        btx = btx.view(n, t, self.integrate_channels, h, w).contiguous()
        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_btx = torch.cat((btx[:, 0:1, :, :, :], btx[:, :-1, :, :, :]), dim=1)
        shifted_btx = shifted_btx.view(n * t, self.integrate_channels, h, w).contiguous()

        atx = atx.view(n, t, self.integrate_channels, h, w).contiguous()
        # Concatenate the zero_frame at the beginning with the shifted_frames
        shifted_atx = torch.cat((atx[:, 1:, :, :, :], atx[:, t - 1:t, :, :, :]), dim=1)
        shifted_atx = shifted_atx.view(n * t, self.integrate_channels, h, w).contiguous()

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
        #     # This assertion block might be slow during validation.
        #     # It can be commented out for performance if confident in the logic.
        #     btx = btx.view(n* t, self.integrate_channels, h, w).contiguous()
        #     atx = atx.view(n* t, self.integrate_channels, h, w).contiguous()
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
        # output_batch = torch.clamp(output_batch, 0., 1.)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds, None, None


if __name__ == '__main__':

    model = WGEN9VSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")

    # model.train() # Set to training mode to test collapsible block logic

    # # Make test run
    # # Example for training mode shape
    # prediction, _, _ = model(torch.randn(2, 64, 64, 30)) # N, H, W, T*C
    # print(prediction.shape)

    # model.eval() # Set to evaluation mode
    # # Example for inference mode shape
    # prediction, _, _ = model(torch.randn(1, 10, 3, 180, 320)) # N, T, C, H, W
    # print(prediction.shape)

