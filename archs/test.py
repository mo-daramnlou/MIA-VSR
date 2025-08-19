import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch
timing=True

@ARCH_REGISTRY.register()
class TST(nn.Module):
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
        super(TST, self).__init__()
        self.scale = scale

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, 2*mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # T convs
        self.tconv1 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1)
        self.tconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # H convs
        self.hconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(2 * mid_channels + 0, out_channels * (scale**2), kernel_size=1)

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
        
        if timing:
          start_event_full = torch.cuda.Event(enable_timing=True)
          end_event_full = torch.cuda.Event(enable_timing=True)
          start_event = torch.cuda.Event(enable_timing=True)
          end_event = torch.cuda.Event(enable_timing=True)
          start_event2 = torch.cuda.Event(enable_timing=True)
          end_event2 = torch.cuda.Event(enable_timing=True)
          start_event3 = torch.cuda.Event(enable_timing=True)
          end_event3 = torch.cuda.Event(enable_timing=True)
        """
        Forward pass.
        Note: PyTorch uses (N, C, H, W) channel order, while the TensorFlow
        model used (N, H, W, C). The model is adapted for the PyTorch convention.
        """
        #  print("lqs: ", lqs.shape) # 32, 64, 64, 30 --  1, 3, 720, 1280
        if timing:
          start_event_full.record()
        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w).contiguous() #320, 3, 64, 64

        
        x= self.fea_conv(lqs_batch)
        fused_features=[]
        for i in range(0,t):
            fused_features.append(x[i])
        if timing:
            start_event.record()
        fused_features = torch.cat(fused_features, dim=0)
        # fused_features = torch.stack(fused_features, dim=0) # 10, 56, 64, 64
        print(fused_features.shape)
        if timing:
            end_event.record()
            start_event2.record()
        fused_features = fused_features.view(n * t, -1 , h, w)
        if timing:
            end_event2.record()
        x= self.psconv(fused_features)
        output_batch =self.pixel_shuffle(x)

        # res =[]
        # for i in range(0,t):
        #     x= self.fea_conv(lqs_batch[i])
        #     x= self.psconv(x)
        #     out =self.pixel_shuffle(x)
        #     res.append(out)
        # output_batch = torch.cat(res, dim=0).view(n * t, 3, h * 4, w * 4)

        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        if timing:
          end_event_full.record()
          print("elapsed_time_ms1: ",start_event.elapsed_time(end_event))
          print("elapsed_time_ms2: ",start_event2.elapsed_time(end_event2))
          print("full_elapsed_time_ms: ",start_event_full.elapsed_time(end_event_full))

        return preds


if __name__ == '__main__':

    model = TST(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/rgenvsr_nobatch_v4.tflite")