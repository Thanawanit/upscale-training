import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class UNetDiscriminatorSN(nn.Module):
    """
    Defines a U-Net discriminator with spectral normalization (D-UNet).
    Used in Phase 2 for sharp polish without inducing ringing artifacts.
    """
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection

        # Downsample branch
        self.conv0 = spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1))

        # Upsample branch
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1))

        # Output layers
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Downsample
        x0 = self.lrelu(self.conv0(x))
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))

        # Upsample with skip connections
        x4 = self.lrelu(self.conv4(self.upsample(x3)))
        if self.skip_connection:
            x4 = x4 + x2

        x5 = self.lrelu(self.conv5(self.upsample(x4)))
        if self.skip_connection:
            x5 = x5 + x1

        x6 = self.lrelu(self.conv6(self.upsample(x5)))
        if self.skip_connection:
            x6 = x6 + x0

        out = self.lrelu(self.conv7(x6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)
        return out
