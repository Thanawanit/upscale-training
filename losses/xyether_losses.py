import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Symmetric smooth L1 variant)
    L = sqrt((pred - target)^2 + eps^2)
    """
    def __init__(self, eps=1e-6, loss_weight=1.0):
        super().__init__()
        self.eps2 = eps ** 2
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = torch.sqrt((pred - target) ** 2 + self.eps2).mean()
        return self.loss_weight * loss


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity (MS-SSIM) Loss
    Maintains line sharpness and structural integrity without ringing.
    Loss = 1.0 - MS_SSIM(pred, target)
    """
    def __init__(self, weights=None, levels=5, win_size=11, win_sigma=1.5, loss_weight=1.0):
        super().__init__()
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = torch.FloatTensor(weights)
        self.levels = levels
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.loss_weight = loss_weight

    def _ssim_per_channel(self, X, Y, window):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        channel = X.size(1)
        # separable 2D conv
        mu1 = F.conv2d(X, window, padding=self.win_size // 2, groups=channel)
        mu2 = F.conv2d(Y, window, padding=self.win_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(X * X, window, padding=self.win_size // 2, groups=channel) - mu1_sq, min=1e-8)
        sigma2_sq = torch.clamp(F.conv2d(Y * Y, window, padding=self.win_size // 2, groups=channel) - mu2_sq, min=1e-8)
        sigma12 = F.conv2d(X * Y, window, padding=self.win_size // 2, groups=channel) - mu1_mu2

        cs_map = torch.clamp((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2), min=1e-5, max=1.0)
        ssim_map = torch.clamp(((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map, min=1e-5, max=1.0)

        return ssim_map.mean(dim=[-1, -2]), cs_map.mean(dim=[-1, -2])

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        device = pred.device

        # Create Gaussian window
        g1d = _fspecial_gauss_1d(self.win_size, self.win_sigma).to(device)
        window = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        window = window.expand(c, 1, self.win_size, self.win_size).contiguous()

        weights = self.weights.to(device)
        mcs = []
        cur_pred = pred
        cur_target = target

        for i in range(self.levels):
            ssim_map, cs_map = self._ssim_per_channel(cur_pred, cur_target, window)
            if i < self.levels - 1:
                mcs.append(cs_map)
                cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2, padding=0)
                cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2, padding=0)
            else:
                final_ssim = ssim_map

        # MS-SSIM product
        ms_ssim = final_ssim.pow(weights[-1])
        for i in range(self.levels - 1):
            ms_ssim = ms_ssim * (mcs[i].pow(weights[i]))

        loss = 1.0 - ms_ssim.mean()
        return self.loss_weight * loss


class ColorLuvLoss(nn.Module):
    """
    CIE-Luv / Color Space Loss
    Encourages flat, clean shading (flat-cel anime) and prevents color banding.
    Computes Charbonnier distance in Luv color space.
    """
    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.charbonnier = CharbonnierLoss(eps=eps)

    def _rgb_to_xyz(self, rgb):
        # Assumes rgb in [0, 1]
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        # sRGB gamma companding
        mask = (rgb > 0.04045).float()
        rgb_linear = mask * torch.pow((rgb + 0.055) / 1.055, 2.4) + (1 - mask) * (rgb / 12.92)

        rl = rgb_linear[:, 0:1, :, :]
        gl = rgb_linear[:, 1:2, :, :]
        bl = rgb_linear[:, 2:3, :, :]

        # sRGB to XYZ matrix
        x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
        y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
        return torch.cat([x, y, z], dim=1)

    def _xyz_to_luv(self, xyz):
        # Reference white D65
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        un = (4 * Xn) / (Xn + 15 * Yn + 3 * Zn)
        vn = (9 * Yn) / (Xn + 15 * Yn + 3 * Zn)

        x = xyz[:, 0:1, :, :]
        y = xyz[:, 1:2, :, :]
        z = xyz[:, 2:3, :, :]

        denom = x + 15 * y + 3 * z + 1e-7
        u_prime = (4 * x) / denom
        v_prime = (9 * y) / denom

        y_rel = y / Yn
        mask = (y_rel > 0.008856).float()
        L = mask * (116.0 * torch.pow(torch.clamp(y_rel, min=1e-7), 1.0 / 3.0) - 16.0) + (1 - mask) * (903.3 * y_rel)
        u = 13.0 * L * (u_prime - un)
        v = 13.0 * L * (v_prime - vn)

        # Scale L to [0, 1] and u, v roughly to [-1, 1]
        return torch.cat([L / 100.0, u / 100.0, v / 100.0], dim=1)

    def forward(self, pred, target):
        pred_luv = self._xyz_to_luv(self._rgb_to_xyz(pred))
        target_luv = self._xyz_to_luv(self._rgb_to_xyz(target))
        return self.loss_weight * self.charbonnier(pred_luv, target_luv)


class FocalFrequencyLoss(nn.Module):
    """
    Fourier Frequency Domain Loss
    Measures frequency discrepancies via 2D Discrete Fourier Transform.
    Directly sharpens high-frequency line art without spatial halos or numerical underflow.
    """
    def __init__(self, loss_weight=0.10, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        loss = torch.abs(pred_freq - target_freq).mean()
        return self.loss_weight * loss


class GANLoss(nn.Module):
    """
    Vanilla GAN Loss for UNet Discriminator
    """
    def __init__(self, loss_weight=0.10):
        super().__init__()
        self.loss_weight = loss_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, d_pred, is_real):
        target = torch.ones_like(d_pred) if is_real else torch.zeros_like(d_pred)
        loss = self.bce(d_pred, target)
        return self.loss_weight * loss


class AdjacentGradientLoss(nn.Module):
    """
    First-Order Adjacent Gradient Loss (No Nyquist Blind Spot).
    Computes exact 1-pixel forward differences in X and Y directions:
      D_x(I) = I[:, :, :, 1:] - I[:, :, :, :-1]
      D_y(I) = I[:, :, 1:, :] - I[:, :, :-1, :]
    Penalizes gradient discrepancies with Charbonnier loss without skipping pixels,
    completely suppressing PixelShuffle checkerboard oscillations while preserving
    acute corners, sharp polygon tips, and subtle object rims.
    """
    def __init__(self, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.charbonnier = CharbonnierLoss(eps=eps)

    def forward(self, pred, target):
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_tar = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_tar = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = self.charbonnier(dx_pred, dx_tar)
        loss_y = self.charbonnier(dy_pred, dy_tar)
        return self.loss_weight * (loss_x + loss_y) * 0.5


class DirectionalGradientLoss(AdjacentGradientLoss):
    """Backward compatibility alias for AdjacentGradientLoss."""
    pass

