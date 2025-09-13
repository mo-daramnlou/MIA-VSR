import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY
# import ai_edge_torch


@ARCH_REGISTRY.register()
class WGEN12VSR(nn.Module):
    def __init__(self, scale=4, in_channels=3, mid_channels=28, num_blocks=4, out_channels=3, integrate_channels=28):
        """
        PyTorch implementation of the base7 TensorFlow model.

        Args:
            scale (int): The upsampling scale factor.
            in_channels (int): Number of channels in the input image.
            num_fea (int): Number of feature channels.
            m (int): Number of middle convolutional layers.
            out_channels (int): Number of channels in the output image.
        """
        super(WGEN12VSR, self).__init__()
        self.scale = scale
        self.integrate_channels=integrate_channels
        self.mid_channels=mid_channels

        # Feature extraction layer
        self.fea_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Middle convolutional layers
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Pre T convs
        self.ptconv = nn.Conv2d(3 * mid_channels, mid_channels, kernel_size=3, padding=1, groups= mid_channels)
        # self.ptconv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups= mid_channels)
        # self.ptconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # T convs
        self.tconv1 = nn.Conv2d(mid_channels , out_channels * (scale**2), kernel_size=1)
        self.tconv2 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(out_channels * (scale**2), out_channels * (scale**2), kernel_size=1)

        # bT convs
        # self.btconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        # self.btconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1, groups= integrate_channels)
        # self.btconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

        # aT convs
        # self.atconv1 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=1)
        # self.atconv2 = nn.Conv2d(mid_channels, integrate_channels, kernel_size=3, padding=1, groups= integrate_channels)
        # self.atconv3 = nn.Conv2d(integrate_channels, integrate_channels, kernel_size=1)

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
        x = self.middle_convs(x)
        x = x + feat_skip

        if self.training:
            x_view = x.view(n, t, self.mid_channels, h, w).contiguous()

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_bx = torch.cat((x_view[:,0:1,:,:,:], x_view[:,:-1,:,:,:]), dim=1)
            shifted_bx = shifted_bx.view(n*t, self.mid_channels, h, w).contiguous()

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_ax = torch.cat((x_view[:,1:,:,:,:], x_view[:,t-1:t,:,:,:]), dim=1)
            shifted_ax = shifted_ax.view(n*t, self.mid_channels, h, w).contiguous()

        else:
            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_bx = torch.cat((x[0:1], x[:-1]), dim=0)

            # Concatenate the zero_frame at the beginning with the shifted_frames
            shifted_ax = torch.cat((x[1:], x[t-1:t]), dim=0)
        

        # Pre-shuffle convolutions
        # if self.training:


        # combined = torch.cat([shifted_bx, x, shifted_ax], dim=1)

        # grouped = combined.view(n*t, 3, self.mid_channels, h, w)
        # transposed = grouped.permute(0,2,1,3,4)
        # conx=transposed.contiguous().view(n*t, 3 * self.mid_channels, h, w)

        # conx = torch.cat((shifted_bx ,x, shifted_ax), dim=2)
        # conx = conx.view(n*t, 3 * self.mid_channels, h, w)
        # else:
        #     conx = torch.cat((shifted_bx.permute(1,0,2,3) ,x.permute(1,0,2,3), shifted_ax.permute(1,0,2,3)), dim=1).reshape(t, 3 * self.mid_channels, h, w)
        #     print(conx.shape)

        # stacked = torch.stack([shifted_bx ,x, shifted_ax], dim=2)
        # # 2. Use .view() to flatten the C and the new dimension together.
        # #    This is a metadata-only operation and is virtually free.
        # #    The -1 tells PyTorch to automatically calculate the new channel size (C*3).
        # #    Shape changes from (N, C, 3, H, W) -> (N, C*3, H, W)
        # conx = stacked.view(n*t, -1, h, w)

        # permuted_list = [t.permute(0, 2, 3, 1).contiguous() for t in [shifted_bx ,x, shifted_ax]]
        # # 2. Concatenate along the LAST dimension (the channel dimension in NHWC).
        # #    This is extremely fast as it just appends memory blocks.
        # #    The final shape is (N, H, W, C*3).
        # interleaved_nhwc = torch.cat(permuted_list, dim=3)
        # # 3. (Optional) Permute back to NCHW if subsequent layers in your PyTorch
        # #    model require it. For TFLite conversion, you can often leave it in NHWC.
        # conx = interleaved_nhwc.permute(0, 3, 1, 2)

        # permuted_list = [t.permute(0, 2, 3, 1) for t in [shifted_bx ,x, shifted_ax]]
        # # 2. Stack the NHWC tensors along a new LAST dimension.
        # #    This groups the corresponding channel values for each pixel.
        # #    Shape changes from (N, H, W, C) -> (N, H, W, C, 3)
        # stacked = torch.stack(permuted_list, dim=4)
        # # 3. View the result to flatten the last two dimensions (C and 3).
        # #    This merges the channels in an interleaved order.
        # #    Shape changes from (N, H, W, C, 3) -> (N, H, W, C*3)
        # conx = stacked.view(n*t, h, w, 3*self.mid_channels).permute(0, 3, 1, 2)

        conx = torch.cat((shifted_bx ,x, shifted_ax), dim=1)


        # print(conx.shape)
        # Assert proper concatenation
        # if self.training:
        x_view = x.view(n* t, self.mid_channels, h, w).contiguous()
        for i,f in enumerate(conx):
            for j, ch in enumerate(f):
                if j%3==0:
                    if i%t == 0:
                        assert torch.equal(f[j], x_view[i,int(j/3)]), ('ass failed1')
                    else: 
                        assert torch.equal(f[j], x_view[i-1,int(j/3)]), ('ass faile2')
                elif j%3==1:
                    assert torch.equal(f[j], x_view[i,int(j/3)]), ('ass failed3')
                else:
                    if i%t == t-1:
                        assert torch.equal(f[j], x_view[i,int(j/3)]), ('ass failed4')
                    else:
                        assert torch.equal(f[j], x_view[i+1,int(j/3)]), ('ass failed5')


        x = self.relu(self.ptconv(conx))
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


if __name__ == '__main__':

    model = WGEN12VSR(mid_channels=28, num_blocks=4)
    model.eval()

    # Make test run
    prediction = model(torch.randn(1, 180, 320, 30))
    print(prediction.shape)

    # Converting model to TFLite

    sample_input = (torch.randn(1, 180, 320, 30),)

    # edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    # edge_model.export("/content/MIA-VSR/assets/genvsr_wo_reshape_triplet8.tflite")