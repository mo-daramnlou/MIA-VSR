import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class RGEN4GRUVSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3):
        super(RGEN4GRUVSR, self).__init__()
        self.scale = scale
        self.mid_channels = mid_channels

        # Feature extraction layer (Unchanged)
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers (Unchanged)
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # --- MODIFICATION START ---
        # 1. Removed the manual aggregation convs (aconv1, aconv2, aconv3)
        # 2. Added an optimized, bidirectional GRU layer
        self.recurrent_core = nn.GRU(
            input_size=mid_channels,
            hidden_size=mid_channels,
            num_layers=1,
            batch_first=True,  # Makes tensor manipulation easier
            bidirectional=True # Matches your original bidirectional logic
        )
        # --- MODIFICATION END ---

        # This layer is well-suited to process the bidirectional GRU output (Unchanged)
        self.ptconv = nn.Conv2d(2 * mid_channels, 2 * mid_channels, kernel_size=3, padding=1, groups=2)

        # T convs (Unchanged)
        self.tconv1 = nn.Conv2d(2 * mid_channels, out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # Pre-shuffle convolutional layers (Unchanged)
        self.psconv = nn.Conv2d(out_channels * (scale**2) + 3, out_channels * (scale**2), kernel_size=1)

        # PixelShuffle layer (Unchanged)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Activation (Unchanged)
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights (Unchanged)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights similar to the Keras version."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Properly initialize GRU weights
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, lqs):
        is_train_mode = len(lqs.shape) == 4
        if is_train_mode:
            n, h, w, tc = lqs.shape
            lqs = lqs.view(n, h, w, -1, 3).permute(0, 3, 4, 1, 2).contiguous()

        n, t, c, h, w = lqs.shape
        lqs_batch = lqs.view(n * t, c, h, w)

        image_skip = lqs_batch
        # 1. Feature extraction (Unchanged)
        x = self.relu(self.fea_conv(lqs_batch))
        feat_skip = x
        x = self.middle_convs(x)
        x = x + feat_skip

        # --- MODIFICATION START ---
        # 2. Bidirectional recurrent aggregation with nn.GRU
        # Reshape for GRU: Treat each pixel's timeline as a sequence.
        # (N*T, C, H, W) -> (N, T, C, H, W)
        x = x.view(n, t, self.mid_channels, h, w)
        # (N, T, C, H, W) -> (N, H, W, T, C)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        # (N, H, W, T, C) -> (N*H*W, T, C) for the GRU
        x = x.view(n * h * w, t, self.mid_channels)

        # Process the entire sequence in one optimized call
        # The output features will have 2 * mid_channels because bidirectional=True
        fused_features, _ = self.recurrent_core(x)

        # Reshape back to image format
        # (N*H*W, T, C*2) -> (N, H, W, T, C*2)
        fused_features = fused_features.view(n, h, w, t, self.mid_channels * 2)
        # (N, H, W, T, C*2) -> (N, T, C*2, H, W)
        fused_features = fused_features.permute(0, 3, 4, 1, 2).contiguous()
        # (N, T, C*2, H, W) -> (N*T, C*2, H, W)
        fused_features = fused_features.view(n * t, self.mid_channels * 2, h, w)
        x = fused_features
        # --- MODIFICATION END ---

        # 3. Upsample the aggregated features (Unchanged)
        x = self.relu(self.ptconv(x))
        x = self.relu(self.tconv1(x))
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))

        x = self.relu(self.psconv(torch.cat((x, image_skip), dim=1)))

        output_batch = self.pixel_shuffle(x)

        # --- Output Shape Handling --- (Unchanged)
        _, c_out, h_out, w_out = output_batch.shape
        preds = output_batch.view(n, t, c_out, h_out, w_out)

        if is_train_mode:
            preds = preds.permute(0, 3, 4, 1, 2).contiguous().view(n, h_out, w_out, t * c_out)

        # The original model returned three values, so we match that format.
        return preds, None,None
    

if __name__ == '__main__':

  model = RGEN4GRUVSR(mid_channels=28, num_blocks=4)
  model.eval()

  # Make test run
  prediction = model(torch.randn(2, 180, 320, 30))
  print(prediction.shape)

  # Converting model to TFLite

  sample_input = (torch.randn(1, 180, 320, 30),)

  # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
  # edge_model.export("/content/MIA-VSR/assets/rgenvsr_nobatch_v4.tflite")