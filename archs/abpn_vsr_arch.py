import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


@ARCH_REGISTRY.register()
class ABPNVSR(nn.Module):
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
        super(ABPNVSR, self).__init__()
        self.scale = scale

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre-shuffle convolutional layers
        self.pre_shuffle_conv1 = nn.Conv2d(mid_channels, out_channels * (scale**2), kernel_size=3, padding=1)
        self.pre_shuffle_conv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

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
        
        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w).contiguous() #320, 3, 64, 64

        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64

        upsampled_inp = torch.cat([lqs_batch] * (self.scale**2), dim=1)

        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        
        # Middle convolutions
        x = self.middle_convs(x)
        
        # Pre-shuffle convolutions
        x = self.relu(self.pre_shuffle_conv1(x))
        x = self.pre_shuffle_conv2(x)
        
        # Add skip connection (equivalent to tf.keras.layers.Add)
        x = x + upsampled_inp

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x)
        
        # Clip the output to a valid image range
        # if not self.training:
        #     output_batch = torch.clamp(output_batch, 0., 255.)

        
         # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds


if __name__ == '__main__':

    model = ABPNVSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/effvsr30.tflite")