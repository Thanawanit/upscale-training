import torch
import torch.nn as nn
import torch.nn.functional as F

class XyetherCompactNet(nn.Module):
    """
    Xyether Strong Series (v1.5 / v2.5 / v3)
    SRVGGNetCompact topology with Global Nearest Residual learning.
    PyTorch Trainable parameters: 600,652 (Identical to net_g_89000 state_dict)
    ONNX graph total initializers: 600,658 (+6 constants for interpolation/clamping bounds)
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.num_feat = num_feat
        self.num_conv = num_conv

        self.body = nn.Sequential()
        # Conv 1: Pre-emphasis Feature Extractor
        self.body.add_module('0', nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.add_module('1', nn.PReLU(num_parameters=num_feat, init=0.2))

        # Body Convs (Conv 2 to 17)
        for i in range(num_conv):
            self.body.add_module(str(2 * i + 2), nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.add_module(str(2 * i + 3), nn.PReLU(num_parameters=num_feat, init=0.2))

        # Conv 18: Reconstruction Head (num_feat -> num_out_ch * upscale^2)
        self.body.add_module(str(2 * num_conv + 2), nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.body.add_module(str(2 * num_conv + 3), nn.PixelShuffle(upscale))

    def forward(self, x):
        # 1. Base Branch: Nearest Neighbor interpolation eliminates sinc-ringing and white halos (Overshoot = 0.00)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')

        # 2. Residual Feature Branch: CNN only computes Anti-Aliasing and flat shading denoise delta
        residual = self.body(x)

        # 3. Combine and clamp to valid optical range
        return torch.clamp(base + residual, 0.0, 1.0)


class XyetherBalancedNet(nn.Module):
    """
    Xyether Balanced 2x (UltraCompact)
    SRVGGNetUltraCompact topology with Global Nearest Residual learning.
    Total parameters: 304,722 (8 body convs)
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=8, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.num_feat = num_feat
        self.num_conv = num_conv

        self.body = nn.Sequential()
        # Conv 1
        self.body.add_module('0', nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.add_module('1', nn.PReLU(num_parameters=num_feat, init=0.2))

        # Body Convs (Conv 2 to 9)
        for i in range(num_conv):
            self.body.add_module(str(2 * i + 2), nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.add_module(str(2 * i + 3), nn.PReLU(num_parameters=num_feat, init=0.2))

        # Conv 10: Reconstruction Head
        self.body.add_module(str(2 * num_conv + 2), nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.body.add_module(str(2 * num_conv + 3), nn.PixelShuffle(upscale))

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        residual = self.body(x)
        return torch.clamp(base + residual, 0.0, 1.0)
