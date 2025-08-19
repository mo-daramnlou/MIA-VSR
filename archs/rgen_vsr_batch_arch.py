#BATCH RGENVSR


import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
#import ai_edge_torch


# @ARCH_REGISTRY.register()
class RGENVSR(nn.Module):
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
        super(RGENVSR, self).__init__()
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
        self.tconv1 = nn.Conv2d(mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # H convs
        self.hconv = nn.Conv2d(out_channels * (scale**2), mid_channels, kernel_size=1)

        # Pre-shuffle convolutional layers
        self.psconv = nn.Conv2d(2 * out_channels * (scale**2) + 0, out_channels * (scale**2), kernel_size=1)

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
        #  print("lqs: ", lqs.shape) # 32, 64, 64, 30 --  1, 3, 720, 1280

        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()
        
        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w).contiguous() #320, 3, 64, 64

        # print("lqs: ", lqs.shape) #32, 10, 3, 64, 64

        # image_skip = lqs_batch
        # Feature extraction
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip=x
        
        # Middle convolutions
        x = self.middle_convs(x)
        x = x + feat_skip







        x = x.view(n, t, -1 , h, w)
        # 3. Bidirectional recurrent aggregation.
        res = []
        
        # Initialize the hidden state. It will contain forward states for the whole batch,
        # followed by backward states for the whole batch.
        # now_frame_forward = x[:, 0, ...]    # Shape: (B, C_feat, h, w)
        # now_frame_backward = x[:, t-1, ...] # Shape: (B, C_feat, h, w)
        # hidden = torch.cat([now_frame_forward, now_frame_backward], dim=0) # Shape: (2*B, C_feat, h, w)
        # res.append(hidden)
        hidden=None

        now_frame_forward = x[:, 0, ...]
        now_frame_backward = x[:, t-1, ...]
        now_frame = torch.cat([now_frame_forward, now_frame_backward], dim=0)
        # T convs
        out = self.relu(self.tconv1(now_frame, dim=1))
        out = self.relu(self.tconv2(out))
        out = self.relu(self.tconv3(out))
        hidden = self.relu(self.hconv(out))
        res.append(out)
        
        for i in range(1, t):
            now_frame_forward = x[:, i, ...]
            now_frame_backward = x[:, t-1-i, ...]
            now_frame = torch.cat([now_frame_forward, now_frame_backward], dim=0) # Shape: (2*B, C_feat, h, w)
            
            # The input to aggr has 2*C_feat channels from hidden and 2*C_feat from now_frame,
            # but they are concatenated along the batch dim, so the channel dim is C_feat + C_feat.
            # The input to aggr is (2*B, 2*C_feat, h, w). This works as conv layers are batch-aware.
            # hidden = self.aggr(torch.cat([hidden, now_frame], dim=1))

            if i == 0:
                hidden= now_frame

            # T convs
            out = self.relu(self.tconv1(torch.cat([hidden, now_frame], dim=1)))
            out = self.relu(self.tconv2(out))
            out = self.relu(self.tconv3(out))
            hidden = self.relu(self.hconv(out))
            res.append(out)
            
        # 4. Upsample the aggregated features.
        # res2 = []
        fused_features=[]
        for i in range(t):
            # For each frame i, fuse the forward features from step i and backward features from step T-1-i.
            fwd_states = res[i][:n, :, :, :]
            bwd_states = res[t-1-i][n:, :, :, :]
            
            fused_features.append(torch.cat([fwd_states, bwd_states], dim=1)) # Shape: (B, 2*C_feat, h, w)
            
        
        # Upsample the batch of fused features.
        # upsampled_frame = self.upsample([fused_features, h, w]) # Shape: (B, 3, 4h, 4w)
        # Pre-shuffle convolutions
        fused_features = torch.stack(fused_features, dim=1).view(n * t, -1 , h, w)
        # fused_features = torch.cat((fused_features, image_skip), dim=1)
        fused_features = self.relu(self.psconv(fused_features))

        # Pixel-Shuffle and final output processing
        output_batch = self.pixel_shuffle(fused_features)

        # res2.append(upsampled_frame)
        
        # 5. Assemble the final output.
        # Stack the upsampled frames along a new time dimension.
        # output_batch = torch.stack(res2, dim=1) # Shape: (B, T, 3, 4h, 4w)
        
        # Reshape to the final output format.
        # output_batch = output_batch.view(n * t, 3, h * 4, w * 4)



        
       

        
        
        # Clip the output to a valid image range
        output_batch = torch.clamp(output_batch, max = 255.)


        # --- Output Shape Handling ---
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)
        # print("preds: ", preds.shape) #32, 256, 256, 30
        
        return preds



if __name__ == '__main__':

    model = RGENVSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 9))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 3),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/rgenvsr_1_improvedpxshuffle_noimgskip.tflite")