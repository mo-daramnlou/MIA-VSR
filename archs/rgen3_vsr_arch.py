import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch

timing=False


@ARCH_REGISTRY.register()
class RGEN3VSR(nn.Module):
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
        super(RGEN3VSR, self).__init__()
        self.scale = scale

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # T convs
        self.tconv1 = nn.Conv2d(3 * mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # H convs
        # self.hconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # Aggr convs
        self.aconv1 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1)
        self.aconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

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

        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64
        if timing:
          start_event.record()
        image_skip = lqs_batch
        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip=x
        
        # Middle convolutions
        x = self.middle_convs(x)
        x = x + feat_skip
        if timing:
          end_event.record()


        if timing:
          start_event2.record()
        #3. Bidirectional recurrent aggregation.
        x = x.view(n, t, -1 , h, w)
        res = []
        
        # Initialize the hidden state for forward and backward passes.
        now_frame_forward = x[:, 0, ...]
        now_frame_backward = x[:, t-1, ...]
        hidden = torch.cat([now_frame_forward, now_frame_backward], dim=0) # Shape: [2, 2*hidden, H, W]
        res.append(hidden)
        # hidden=None
        for i in range(1, t):
            res.append(hidden)
            if i == t-1:
               continue
            # Get current frames for both directions.
            now_frame_forward = x[:, i, ...]
            now_frame_backward = x[:, t-1-i, ...]
            now_frame = torch.cat([now_frame_forward, now_frame_backward], dim=0)
            
            hidden = self.relu(self.aconv1(torch.cat([hidden, now_frame], dim=1)))
            hidden = self.relu(self.aconv2(hidden))

        if timing:
          end_event2.record()
          start_event3.record()
        # 4. Upsample the aggregated features.
        res2 = []
        fused_features=[]
        
        for i in range(0, t):
            # t = []
            # Fuse the forward and backward features for the current time step along the channel dimension.
            
            # Upsample the fused features. The input to upsample is a batch created by concatenating items in t.
            # t_0_1_res = self.upsample([t, h, w])
            # print("res:", res[i].shape)
            fused_features.append(torch.cat([res[i][:n, :, :, :], x[:,i], res[t-1-i][n:, :, :, :]], dim=1))

        # print(fused_features[0].shape)
        # print(fused_features[1].shape)
        # fused_features = torch.cat(fused_features, dim=0).view(n * t, -1 , h, w) # 10, 56, 64, 64
        fused_features = torch.stack(fused_features, dim=1).contiguous().view(n * t, -1, h, w)
        # 1. Concatenate along the batch dimension (groups by time)
        # x = torch.cat(fused_features, dim=0)

        # # 2. Reshape to expose time and batch dims, then permute them
        # #    This changes the order from (t, n, c, h, w) to (n, t, c, h, w)
        # x = x.view(t, n, *x.shape[1:])
        # x = x.permute(1, 0, 2, 3, 4)

        # # 3. Flatten to the final desired shape, now with the correct order
        # fused_features = x.contiguous().view(n * t, -1, h, w)
        
        x = fused_features
        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
          # Pre-shuffle convolutions
        # print(x.shape)
        # print(image_skip.shape)
        x = self.relu(self.psconv(torch.cat((x, image_skip), dim=1)))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(x)
        
        # res2.append(out)
        
        # Concatenate the list of upsampled frames into a single tensor.
        # The result is the learned residual.
        # output_batch = torch.cat(res2, dim=0).view(n * t, 3, h * 4, w * 4)
        
        # 5. Add the learned residual to the bilinear upsampling result.
        # return residual



       

       
        
        # Clip the output to a valid image range
        # output_batch = torch.clamp(output_batch, max = 255.)
        if timing:
          end_event3.record()
          elapsed_time_ms = start_event.elapsed_time(end_event)
          print("elapsed_time_ms1: ",elapsed_time_ms)
          elapsed_time_ms = start_event2.elapsed_time(end_event2)
          print("elapsed_time_ms2: ",elapsed_time_ms)
          elapsed_time_ms = start_event3.elapsed_time(end_event3)
          print("elapsed_time_ms3: ",elapsed_time_ms)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        if timing:
          end_event_full.record()
          print("full_elapsed_time_ms: ",start_event_full.elapsed_time(end_event_full))

        return preds, None, None

if __name__ == '__main__':

    model = RGEN3VSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(2, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/rgenvsr_nobatch_v4.tflite")