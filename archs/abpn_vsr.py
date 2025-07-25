import torch
import torch.nn as nn
import math

class ABPNVSR(nn.Module):
    def __init__(self, scale=3, in_channels=3, num_fea=28, m=4, out_channels=3):
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
        self.fea_conv = nn.Conv2d(in_channels, num_fea, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(m):
            middle_layers.append(nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre-shuffle convolutional layers
        self.pre_shuffle_conv1 = nn.Conv2d(num_fea, out_channels * (scale**2), kernel_size=3, padding=1)
        self.pre_shuffle_conv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)

        # PixelShuffle layer (equivalent to tf.nn.depth_to_space)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    #     # Initialize weights
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """Initializes weights similar to the Keras version."""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # glorot_normal initializer in Keras is Xavier normal in PyTorch
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 # bias_initializer='zeros'
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        Note: PyTorch uses (N, C, H, W) channel order, while the TensorFlow
        model used (N, H, W, C). The model is adapted for the PyTorch convention.
        """
        # The upsampling in TensorFlow is a skip connection after channel-wise repeat.
        # This is equivalent to adding it before the final pixel shuffle.
        # tf.concat([inp]*(scale**2), axis=3) -> torch.cat([x]*(scale**2), dim=1)
        # We need to ensure in_channels == out_channels for the addition to work,
        # which is true by default (3).
        if self.in_channels == self.out_channels:
            upsampled_inp = torch.cat([x] * (self.scale**2), dim=1)
        else:
            # Handle case where in_channels != out_channels if necessary,
            # though the original model assumes they are equal.
            # For this conversion, we strictly adhere to the original logic.
            raise ValueError("in_channels must equal out_channels for the skip connection.")


        # Feature extraction
        x = self.relu(self.fea_conv(x))
        
        # Middle convolutions
        x = self.middle_convs(x)
        
        # Pre-shuffle convolutions
        x = self.relu(self.pre_shuffle_conv1(x))
        x = self.pre_shuffle_conv2(x)
        
        # Add skip connection (equivalent to tf.keras.layers.Add)
        x = x + upsampled_inp

        # Pixel-Shuffle and final output processing
        out = self.pixel_shuffle(x)
        
        # Clip the output to a valid image range
        out = torch.clamp(out, 0., 255.)
        
        return out
    
    # Properties to match constructor arguments for clarity
    @property
    def in_channels(self):
        return self.fea_conv.in_channels

    @property
    def out_channels(self):
        return self.pixel_shuffle.out_channels


if __name__ == '__main__':
    # Instantiate the model with default parameters
    model = ABPNVSR()
    model.eval() # Set model to evaluation mode

    # Calculate and print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'PyTorch Model instantiated.')
    print('Params: [{:.2f}]K'.format(total_params / 1e3))

    # Test with a dummy input tensor
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    print(f'Dummy Input Shape: {dummy_input.shape}')
    print(f'Output Shape: {output.shape}')