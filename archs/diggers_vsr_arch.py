import torch
import torch.nn as nn
import torch.nn.functional as F
# import ai_edge_torch

class SEL(nn.Module):
    """
    PyTorch implementation of the SEL (Squeeze-and-Excitation-Like) block.
    This block performs channel-wise attention.
    """
    def __init__(self, hidden):
        super(SEL, self).__init__()
        # Convolutional layer to compute the attention map.
        # padding='same' in TF with kernel 3 is padding=1 in PyTorch.
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Output tensor after applying channel attention.
        """
        # The attention mechanism: input is multiplied by a sigmoid-activated attention map.
        return x * torch.sigmoid(self.conv(self.relu(x)))

class Upsample(nn.Module):
    """
    PyTorch implementation of the Upsample module.
    This module refines features and upscales them by 4x.
    """
    def __init__(self, hidden):
        super(Upsample, self).__init__()
        # Shrinking convolution to adjust channel dimensions from the fused input.
        # It takes 2*hidden channels (from the fused forward/backward passes) and shrinks to hidden channels.
        self.shirking = nn.Conv2d(2 * hidden, hidden *3, kernel_size=1, stride=1, padding=0)
        self.sel = SEL(hidden *3)
        self.reconstruction = IMDModule(in_channels=hidden)

        # HR convolutions with LeakyReLU activation. These operate on `hidden` channels.
        self.conv_hr1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_hr2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Final convolution to produce the 3-channel (RGB) output.
        self.conv_last = nn.Conv2d(hidden, 3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): A tuple containing the feature tensor, target height, and target width.
        Returns:
            torch.Tensor: The upscaled RGB image tensor.
        """
        out, h, w = inputs
        out = self.sel(self.shirking(out))
        # out = self.reconstruction(out)

        # # First upsampling step (2x) using bilinear interpolation.
        # out = F.interpolate(out, size=(h * 2, w * 2), mode='bilinear', align_corners=False)
        # out = self.conv_hr1(out)
        
        # # Second upsampling step (another 2x, for a total of 4x)
        # out = F.interpolate(out, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        # out = self.conv_hr2(out)
        
        # out = self.conv_last(out)
        # return out
        return self.pixel_shuffle(out)

class IMDModule(nn.Module):
    """
    PyTorch implementation of the Information Multi-distillation Module (IMDModule).
    This module uses channel splitting and feature fusion to extract rich features.
    """
    def __init__(self, in_channels, c1=20, c2=12, c3=4, distillation_rate=0.5):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        
        # Convolutional blocks with LeakyReLU activation.
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(c1 - self.distilled_channels, c2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(c2 - self.distilled_channels, c3, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Fusion convolution. It takes the concatenated features as input.
        # The number of input channels is the sum of all distilled channels, the final block's output, and the original input channels.
        concat_channels = self.distilled_channels * 2 + c3 + in_channels
        self.c5 = nn.Conv2d(concat_channels, in_channels, kernel_size=1, padding=0)
        self.sel = SEL(in_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature tensor.
        Returns:
            torch.Tensor: Output tensor after feature distillation and fusion.
        """
        out_c1 = self.c1(x)
        # Distill: Split channels into two groups.
        distilled_c1 = out_c1[:, :self.distilled_channels, :, :]
        remaining_c1 = out_c1[:, self.distilled_channels:, :, :]

        out_c2 = self.c2(remaining_c1)
        distilled_c2 = out_c2[:, :self.distilled_channels, :, :]
        remaining_c2 = out_c2[:, self.distilled_channels:, :, :]

        out_c3 = self.c3(remaining_c2)
        
        # Concatenate all distilled features, the final feature block, and the original input.
        # In PyTorch, concatenation is along dimension 1 (channels).
        out = torch.cat([distilled_c1, distilled_c2, out_c3, x], dim=1)
        
        out_fused = self.sel(self.c5(out))
        return out_fused

def get_bilinear(image, scale_factor=4):
    """
    Upscales an image using bilinear interpolation.
    
    Args:
        image (torch.Tensor): The input image tensor (N, C, H, W).
        scale_factor (int): The factor by which to upscale.
    Returns:
        torch.Tensor: The upscaled image.
    """
    return F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)

class DIGVSR(nn.Module):
    """
    PyTorch implementation of the main BidirectionalRestorer_V6 model.
    """
    def __init__(self, hidden_channels=8):
        super(DIGVSR, self).__init__()
        self.hidden_channels = hidden_channels
        
        # Initial feature extraction from RGB frames.
        # padding='same' in TF with kernel 5 is padding=2 in PyTorch.
        self.conv1 = nn.Conv2d(3, hidden_channels, kernel_size=5, stride=1, padding=2)
        
        # Feature extractors for the RGB and aggregated features.
        self.feature_extracter_rgb1 = IMDModule(in_channels=hidden_channels)
        self.feature_extracter_rgb2 = IMDModule(in_channels=hidden_channels)
        self.feature_extracter_aggr1 = IMDModule(in_channels=2 * hidden_channels)
        
        # Shrinking convolution for the aggregation step.
        self.shirking1 = nn.Conv2d(4 * hidden_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0)
        
        # The final upsampling module.
        self.upsample = Upsample(2 * hidden_channels)

    def rgb(self, x):
        """Processes the initial RGB frames to extract features."""
        x = self.conv1(x)
        x1 = self.feature_extracter_rgb1(x)
        x2 = self.feature_extracter_rgb2(x1)
        # Concatenate features from two stages.
        return torch.cat([x1, x2], dim=1)

    def aggr(self, x):
        """Aggregates features from the current and previous time steps."""
        x = self.shirking1(x)
        x1 = self.feature_extracter_aggr1(x)
        return x1

    def forward(self, inputs):
        start_event_full = torch.cuda.Event(enable_timing=True)
        end_event_full = torch.cuda.Event(enable_timing=True)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event2 = torch.cuda.Event(enable_timing=True)
        end_event2 = torch.cuda.Event(enable_timing=True)
        start_event3 = torch.cuda.Event(enable_timing=True)
        end_event3 = torch.cuda.Event(enable_timing=True)
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, T*C, H, W), e.g., (1, 30, 180, 320).
        Returns:
            torch.Tensor: The super-resolved output tensor of shape (N, T*C, 4*H, 4*W).
        """
        # Note: PyTorch uses NCHW format, so channels come before height and width.
        # The input is expected as [1, 30, 180, 320] for 10 frames.
        start_event_full.record()
        B, C_total, h, w = inputs.shape
        T = int(C_total/3) # Number of frames
        
        start_event.record()
        # 1. Get the coarse bilinear upsampling result first.
        biup = get_bilinear(inputs, scale_factor=4)
        
        # 2. Reshape and process initial RGB features.
        # Reshape from (B, T*C, H, W) to (B*T, C, H, W)
        inputs = inputs.view(B, T, 3, h, w).view(B * T, 3, h, w)
        inputs = self.rgb(inputs) # Output shape: [10, 2*hidden, H, W]
        end_event.record()
        start_event2.record()
        # 3. Bidirectional recurrent aggregation.
        res = []
        
        # Initialize the hidden state for forward and backward passes.
        now_frame_forward = inputs[0:1, :, :, :]
        now_frame_backward = inputs[T-1:T, :, :, :]
        hidden = torch.cat([now_frame_forward, now_frame_backward], dim=0) # Shape: [2, 2*hidden, H, W]
        res.append(hidden)
        # hidden = None
        for i in range(1, T):
            # Get current frames for both directions.
            now_frame_forward = inputs[i:i+1, :, :, :]
            now_frame_backward = inputs[(T-i-1):(T-i), :, :, :]
            now_frame = torch.cat([now_frame_forward, now_frame_backward], dim=0)
            
            # if i == 0:
            #     hidden = now_frame
            # Concatenate previous hidden state with current frame features.
            # Shape becomes [2, 4*hidden, H, W] before aggregation.
            hidden = self.aggr(torch.cat([hidden, now_frame], dim=1))
            res.append(hidden)
        end_event2.record()
        start_event3.record()
        # 4. Upsample the aggregated features.
        res2 = []
        for i in range(0, T):
            # t = []
            # Fuse the forward and backward features for the current time step along the channel dimension.
            t = torch.cat([res[i][0:1, :, :, :], res[T-i-1][1:2, :, :, :]], dim=1)
            
            # Upsample the fused features. The input to upsample is a batch created by concatenating items in t.
            t_0_1_res = self.upsample([t, h, w])
            
            res2.append(t_0_1_res)
        
        # Concatenate the list of upsampled frames into a single tensor.
        # The result is the learned residual.

        end_event3.record()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print("elapsed_time_ms1: ",elapsed_time_ms)
        elapsed_time_ms = start_event2.elapsed_time(end_event2)
        print("elapsed_time_ms2: ",elapsed_time_ms)
        elapsed_time_ms = start_event3.elapsed_time(end_event3)
        print("elapsed_time_ms3: ",elapsed_time_ms)

        residual = torch.cat(res2, dim=0).view(B, T * 3, h * 4, w * 4)
        
        # 5. Add the learned residual to the bilinear upsampling result.
        end_event_full.record()
        print("full_elapsed_time_ms: ",start_event_full.elapsed_time(end_event_full))
        return residual
        # return torch.clamp(residual + biup, 0, 1)

if __name__ == '__main__':
    
    model = DIGVSR()
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 3, 180, 320))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 30, 180, 320),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/effvsr30.tflite")
