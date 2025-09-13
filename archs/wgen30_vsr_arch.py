import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
# from basicsr.utils.registry import ARCH_REGISTRY
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
        x = self.block(inputs)
        print("x: ",x.shape)
        return x


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

        # If num_inner_layers > 1, then use another conv1x1 at the beginning

        # --- Learnable Convs for Collapsing ---
        # These layers will operate on the small 'delta' tensor to generate
        # the collapsed weights. These are the original convs from LinearBlock_e.
        self.conv_expand = nn.Conv2d(in_channels=self.in_filters,
                                     out_channels=self.feature_size,
                                     kernel_size=(self.ky, self.kx),
                                     padding='valid') # Padding is handled by F.conv2d later
        self.conv_project = nn.Conv2d(in_channels=self.feature_size,
                                      out_channels=self.out_filters,
                                      kernel_size=1,
                                      padding='valid')

        self.collapsed_weights = None

        # --- Residual Tensor ---
        # This tensor represents the identity connection (skip connection) that will be
        # arithmetically added to the generated weights.
        # residual = torch.zeros(self.out_filters, self.in_filters, self.ky, self.kx)
        # if self.in_filters == self.out_filters:
        #     mid_kx = self.kx // 2
        #     mid_ky = self.ky // 2
        #     for i in range(self.out_filters):
        #         residual[i, i, mid_ky, mid_kx] = 1.0
        # self.register_buffer('residual', residual)

    def forward(self, inputs):
        # The 'self.training' flag is a built-in PyTorch attribute from nn.Module.
        # It's True during training (.train() mode) and False during evaluation (.eval() mode).
        # if self.training or self.collapsed_weights is None:
        # --- Online Linear Collapse ---
        # The expanded block is Conv_1x1(Conv_3x3(input)).
        # To fuse them, we convolve the 1x1 kernel over the 3x3 kernel.
        # conv_expand.weight: [feature_size, in_filters, 3, 3]
        # conv_project.weight: [out_filters, feature_size, 1, 1]
        # We can treat the 3x3 kernels as a batch of images and the 1x1 kernels as the conv weights.
        w_3x3 = self.conv_expand.weight
        w_1x1 = self.conv_project.weight
        wt_tensor = F.conv2d(w_3x3.transpose(0, 1), w_1x1, padding='same').transpose(0, 1)

        # --- Final Convolution ---
        return F.conv2d(inputs, wt_tensor, stride=1, padding="same")


# @ARCH_REGISTRY.register()
class WGEN30VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=24, num_blocks=6, out_channels=3, integrate_channels=16, expand_size=120):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN30VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(int(num_blocks/2)):
            middle_layers.append(LinearBlock_c(
                in_filters=mid_channels,
                num_inner_layers=1, 
                kernel_size=3,
                padding='same',
                out_filters=mid_channels,
                feature_size=expand_size, 
                mode='train' # The block handles train/eval switching internally
            ))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs1 = nn.Sequential(*middle_layers)
        middle_layers1 = []
        for _ in range(int(num_blocks/2)):
            middle_layers1.append(LinearBlock_c(
                in_filters=mid_channels,
                num_inner_layers=1,  # As per the SESR paper for 3x3 blocks
                kernel_size=3,
                padding='same',
                out_filters=mid_channels,
                feature_size=expand_size, # Internal expansion dimension
                mode='train' # The block handles train/eval switching internally
            ))
            middle_layers1.append(nn.ReLU(inplace=True))
        self.middle_convs2 = nn.Sequential(*middle_layers1)

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
        # self.btconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.btconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.btconv5 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        # self.atconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.atconv4 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=3, padding=1)
        self.atconv5 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

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
        x = self.middle_convs1(x)
        x = self.middle_convs2(x+feat_skip)
        x = x + feat_skip

        # Pre T convs
        ptx = self.relu(self.ptconv2(x))
        ptx = self.relu(self.ptconv3(ptx))
        ptx = self.relu(self.ptconv4(ptx))
        ptx = self.relu(self.ptconv5(ptx))

        # bT convs
        # btx = self.relu(self.btconv1(x))
        btx = self.relu(self.btconv2(x))
        btx = self.relu(self.btconv3(btx))
        btx = self.relu(self.btconv4(btx))
        btx = self.relu(self.btconv5(btx))

        # aT convs
        # atx = self.relu(self.atconv1(x))
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
        
        # Clip the output to a valid image range
        # output_batch = torch.clamp(output_batch, max = 255.)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds
    

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




def convert_weights(trained_model_path, infer_model_path):
    """
    Converts weights from a trained WGEN30VSR model to the WGEN30VSRInfer format.
    """
    # Instantiate the training and inference models
    # Use the same hyperparameters as your trained model
    train_model = WGEN30VSR(mid_channels=24, num_blocks=6, expand_size=120)
    infer_model = WGEN30VSRInfer(mid_channels=24, num_blocks=6)

    # Load the entire checkpoint
    loaded_checkpoint = torch.load(trained_model_path)

    # Extract the actual model parameters from the checkpoint
    if 'params' in loaded_checkpoint:
        train_state_dict = loaded_checkpoint['params']
    elif 'params_ema' in loaded_checkpoint:
        train_state_dict = loaded_checkpoint['params_ema']
    else:
        train_state_dict = loaded_checkpoint

    # --- FIX: Load the state dict with strict=False ---
    # This will ignore the unexpected 'collapsed_weights' keys.
    train_model.load_state_dict(train_state_dict, strict=False)
    train_model.eval()

    # Create a new state dict for the inference model
    infer_state_dict = OrderedDict()

    # Get the state dict from the now-loaded training model
    train_model_state_dict = train_model.state_dict()

    # Iterate through the state dict of the training model to fuse weights
    for key, value in train_model_state_dict.items():
        # Check for LinearBlock_c weights that need to be fused
        if 'middle_convs' in key and 'conv_expand.weight' in key:
            # This is the 3x3 weight of a LinearBlock_c
            # e.g., 'middle_convs1.0.conv_expand.weight'
            
            # Get the corresponding 1x1 projection weight
            proj_key = key.replace('conv_expand.weight', 'conv_project.weight')
            w_1x1 = train_model_state_dict[proj_key]
            w_3x3 = value
            
            # Fuse the weights by convolving the kernels
            fused_weight = F.conv2d(w_3x3.transpose(0, 1), w_1x1, padding='same').transpose(0, 1)
            
            # Get the key for the inference model's standard Conv2d layer
            infer_key = key.replace('.conv_expand.weight', '.weight')
            infer_state_dict[infer_key] = fused_weight
            print(f"Fused {key} -> {infer_key}")

        elif 'middle_convs' in key and ('conv_expand.bias' in key or 'conv_project' in key):
            # Skip the individual biases and the projection weights as they are now fused
            continue
        else:
            # Copy all other weights directly
            infer_state_dict[key] = value

    # Load the new state dict into the inference model
    infer_model.load_state_dict(infer_state_dict)
    print("\nWeight conversion complete.")

    # Save the inference model's state dict
    torch.save(infer_model.state_dict(), infer_model_path)
    print(f"Inference model saved to {infer_model_path}")

if __name__ == '__main__':

    # trained_model_path = '/content/net_g_30000.pth'
    # infer_model_path = '/content/net_g_30000_infer.pth'
    # convert_weights(trained_model_path, infer_model_path)

    model = WGEN30VSRInfer(mid_channels=24, num_blocks=6)
    model.eval()
    model.load_state_dict(torch.load("/content/net_g_30000_infer.pth"), strict=True)

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 9))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/wgen30vsri2.tflite")