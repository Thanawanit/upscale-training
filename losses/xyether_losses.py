import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Symmetric smooth L1 variant)
    L = sqrt((pred - target)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-6, loss_weight: float = 1.0):
        super().__init__()
        self.eps2 = eps ** 2
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt((pred - target) ** 2 + self.eps2).mean()
        return self.loss_weight * loss


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity (MS-SSIM) Loss
    Maintains line sharpness and structural integrity without ringing.
    Loss = 1.0 - MS_SSIM(pred, target)
    """
    def __init__(self, weights=None, levels: int = 5, win_size: int = 11, win_sigma: float = 1.5, loss_weight: float = 1.0):
        super().__init__()
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = torch.FloatTensor(weights)
        self.levels = levels
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.loss_weight = loss_weight

    def _ssim_per_channel(self, X: torch.Tensor, Y: torch.Tensor, window: torch.Tensor):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        channel = X.size(1)
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        device = pred.device

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
    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.charbonnier = CharbonnierLoss(eps=eps)

    def _rgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        mask = (rgb > 0.04045).float()
        rgb_linear = mask * torch.pow((rgb + 0.055) / 1.055, 2.4) + (1 - mask) * (rgb / 12.92)

        rl = rgb_linear[:, 0:1, :, :]
        gl = rgb_linear[:, 1:2, :, :]
        bl = rgb_linear[:, 2:3, :, :]

        x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
        y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
        return torch.cat([x, y, z], dim=1)

    def _xyz_to_luv(self, xyz: torch.Tensor) -> torch.Tensor:
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

        return torch.cat([L / 100.0, u / 100.0, v / 100.0], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_luv = self._xyz_to_luv(self._rgb_to_xyz(pred))
        target_luv = self._xyz_to_luv(self._rgb_to_xyz(target))
        return self.loss_weight * self.charbonnier(pred_luv, target_luv)


class FocalFrequencyLoss(nn.Module):
    """
    Fourier Frequency Domain Loss
    Measures frequency discrepancies via 2D Discrete Fourier Transform.
    """
    def __init__(self, loss_weight: float = 0.10, alpha: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        loss = torch.abs(pred_freq - target_freq).mean()
        return self.loss_weight * loss


class GANLoss(nn.Module):
    """
    Vanilla GAN Loss for UNet Discriminator
    """
    def __init__(self, loss_weight: float = 0.10):
        super().__init__()
        self.loss_weight = loss_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, d_pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(d_pred) if is_real else torch.zeros_like(d_pred)
        loss = self.bce(d_pred, target)
        return self.loss_weight * loss


class VGGPerceptualLoss(nn.Module):
    """
    Shallow-Only VGG19 Perceptual Loss (conv1_2 and conv2_2 ONLY).
    Eliminates deep MaxPool2d stride-4/stride-8 aliasing (conv3_4, conv4_4) that caused
    2x2 PixelShuffle checkerboards and double-line echoes 2-3px away from edges.
    """
    def __init__(self, loss_weight: float = 0.08):
        super().__init__()
        self.loss_weight = loss_weight
        self.slice1 = None
        self.slice2 = None
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _lazy_init_vgg(self, device):
        if self.slice1 is None:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
            for p in vgg.parameters():
                p.requires_grad = False
            self.slice1 = vgg[:4]   # conv1_2 (stride 1, exact 1px alignment!)
            self.slice2 = vgg[4:9]  # conv2_2 (stride 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        self._lazy_init_vgg(device)
        p = (pred - self.mean.to(device)) / self.std.to(device)
        t = (target - self.mean.to(device)) / self.std.to(device)

        p1 = self.slice1(p); t1 = self.slice1(t)
        p2 = self.slice2(p1); t2 = self.slice2(t1)

        loss = 0.70 * F.l1_loss(p1, t1) + 0.30 * F.l1_loss(p2, t2)
        return self.loss_weight * loss


def build_true_ink_trough_target(hr_tensor: torch.Tensor, valley_strength: float = 1.55, aa_preserve: float = 0.18):
    """
    Exact Bilateral-Minimum 1px Non-Maximum Suppression (NMS) Ink-Trough Target:
      InkTrough_d(p) = max(0, min(max(I_{p-1}, I_{p-2}), max(I_{p+1}, I_{p+2})) - I_p) * 1(I_p <= I_{p-1} & I_p <= I_{p+1})
    where I = 0.5 * Luma + 0.5 * max(R, G, B).
    - On step edges without a dark ink line (e.g. 158 -> 105), min(158, 105) - 105 == 0.0000 (zero false darkening).
    - On true 1px ink lines & hairlines (e.g. 158 -> 67 -> 105), min(158, 105) - 67 > 0, darkening ONLY the 1px center
      multiplicatively so brown/colored lines retain their hue and 0.0000 bright halo is introduced.
    """
    r, g, b = hr_tensor[:, 0:1], hr_tensor[:, 1:2], hr_tensor[:, 2:3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    vmax = torch.maximum(torch.maximum(r, g), b)
    illum = 0.5 * luma + 0.5 * vmax

    ip1 = F.pad(illum, (1, 1, 1, 1), mode='replicate')
    ip2 = F.pad(illum, (2, 2, 2, 2), mode='replicate')

    # 1. Horizontal (0 deg)
    l1, r1 = ip1[:, :, 1:-1, :-2], ip1[:, :, 1:-1, 2:]
    l2, r2 = ip2[:, :, 2:-2, :-4], ip2[:, :, 2:-2, 4:]
    is_min_h = ((illum <= l1) & (illum <= r1) & ((illum < l1) | (illum < r1))).float()
    bilat_min_h = torch.minimum(torch.maximum(l1, l2), torch.maximum(r1, r2))
    depth_h = torch.clamp(bilat_min_h - illum, min=0.0) * is_min_h

    # 2. Vertical (90 deg)
    u1, d1 = ip1[:, :, :-2, 1:-1], ip1[:, :, 2:, 1:-1]
    u2, d2 = ip2[:, :, :-4, 2:-2], ip2[:, :, 4:, 2:-2]
    is_min_v = ((illum <= u1) & (illum <= d1) & ((illum < u1) | (illum < d1))).float()
    bilat_min_v = torch.minimum(torch.maximum(u1, u2), torch.maximum(d1, d2))
    depth_v = torch.clamp(bilat_min_v - illum, min=0.0) * is_min_v

    # 3. Diagonal 1 (45 deg)
    tl1, br1 = ip1[:, :, :-2, :-2], ip1[:, :, 2:, 2:]
    tl2, br2 = ip2[:, :, :-4, :-4], ip2[:, :, 4:, 4:]
    is_min_d1 = ((illum <= tl1) & (illum <= br1) & ((illum < tl1) | (illum < br1))).float()
    bilat_min_d1 = torch.minimum(torch.maximum(tl1, tl2), torch.maximum(br1, br2))
    depth_d1 = torch.clamp(bilat_min_d1 - illum, min=0.0) * is_min_d1

    # 4. Diagonal 2 (135 deg)
    tr1, bl1 = ip1[:, :, :-2, 2:], ip1[:, :, 2:, :-2]
    tr2, bl2 = ip2[:, :, :-4, 4:], ip2[:, :, 4:, :-4]
    is_min_d2 = ((illum <= tr1) & (illum <= bl1) & ((illum < tr1) | (illum < bl1))).float()
    bilat_min_d2 = torch.minimum(torch.maximum(tr1, tr2), torch.maximum(bl1, bl2))
    depth_d2 = torch.clamp(bilat_min_d2 - illum, min=0.0) * is_min_d2

    is_min_any = torch.clamp(is_min_h + is_min_v + is_min_d1 + is_min_d2, 0.0, 1.0)
    core_valley = torch.maximum(torch.maximum(depth_h, depth_v), torch.maximum(depth_d1, depth_d2))
    core_valley = torch.where(core_valley > 0.015, core_valley, torch.zeros_like(core_valley))

    k_aa = torch.tensor(
        [[0.05, 0.10, 0.05],
         [0.10, 0.40, 0.10],
         [0.05, 0.10, 0.05]], dtype=hr_tensor.dtype, device=hr_tensor.device
    ).view(1, 1, 3, 3)
    aa_valley = F.conv2d(F.pad(core_valley, (1, 1, 1, 1), mode='replicate'), k_aa)
    shoulder_h = torch.clamp(bilat_min_h - illum, min=0.0)
    shoulder_v = torch.clamp(bilat_min_v - illum, min=0.0)
    is_true_shoulder = ((shoulder_h > 0.015) | (shoulder_v > 0.015)).float()
    effective_valley = core_valley + aa_preserve * aa_valley * is_true_shoulder * (1.0 - is_min_any)

    darken_factor = torch.clamp(1.0 - valley_strength * effective_valley, min=0.15, max=1.0)
    dark_ink_toe = torch.clamp((0.12 - luma) / 0.12, min=0.0, max=1.0) * (core_valley > 0.02).float()
    out = hr_tensor * darken_factor - 0.045 * dark_ink_toe

    return torch.clamp(out, 0.0, 1.0), core_valley


class BilateralInkTroughTarget(nn.Module):
    """
    Self-contained mathematical target generator for Phase 18 (Zero-Halo 1px Core Engine).
    Requires no external teacher networks.
    """
    def __init__(self, valley_strength: float = 1.55, aa_preserve: float = 0.18):
        super().__init__()
        self.valley_strength = valley_strength
        self.aa_preserve = aa_preserve

    @torch.no_grad()
    def forward(self, hr_tensor: torch.Tensor):
        return build_true_ink_trough_target(hr_tensor, self.valley_strength, self.aa_preserve)


class Phase18CoreEngineLoss(nn.Module):
    """
    Phase 18 Zero-Halo 1px Core Engine Loss:
    1. Decoupled Active Subpixel Split vs Quiet Flat Cel-Shading Loss:
       Separately normalizes active edge/hairline subpixel splits (|T - Nearest(LR)| > 0.010)
       from quiet flat regions so the 94% flat background never drowns out diagonal anti-aliasing.
    2. Output Residual RMS Energy Anchor:
       Prevents nullspace collapse in the 64->12 PixelShuffle head and ensures full subpixel
       split dynamic range (ResMax ~ 170-180) on diagonal contours and 1px hairlines.
    3. 1-Pixel Directional Contrast & Diagonal Continuity Loss (X, Y, and 45/135 diagonals).
    4. 8-Neighbor Isotropic Laplacian Loss (12x suppression of 2x2 PixelShuffle checkerboard).
    5. 5x5 Edge-Halo Zone Envelope Clamp (strictly prevents white/cyan halo overshoot).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        lap8 = torch.tensor(
            [[1., 1., 1.],
             [1., -8., 1.],
             [1., 1., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 8.0
        self.register_buffer('lap8_kernel', lap8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lr_input: torch.Tensor, core_valley: torch.Tensor):
        eps = self.eps
        with torch.no_grad():
            base = F.interpolate(lr_input, scale_factor=2, mode='nearest')
            r_tar = target - base
            r_mag = r_tar.abs().mean(dim=1, keepdim=True)

            t_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')
            t_max3 = F.max_pool2d(t_pad, kernel_size=3, stride=1)
            t_min3 = -F.max_pool2d(-t_pad, kernel_size=3, stride=1)
            edge_span = (t_max3 - t_min3).mean(dim=1, keepdim=True)

            is_core = (core_valley > 0.012).float()
            is_edge = (edge_span > 0.030).float()
            is_active = ((r_mag > 0.010) | (is_core > 0.5) | (is_edge > 0.5)).float()
            is_quiet = 1.0 - is_active

            edge_halo_zone = F.max_pool2d(F.pad(is_edge, (2, 2, 2, 2), mode='replicate'), kernel_size=5, stride=1)

            w_active = is_active * (1.0 + 26.0 * r_mag + 12.0 * is_core)
            w_active_norm = w_active / (w_active.mean() + 1e-6)

        r_pred = pred - base
        err_charb = torch.sqrt((r_pred - r_tar) ** 2 + eps ** 2)

        # 1. Active Subpixel Anti-Aliasing & 1px Valley Split Loss
        under_res = F.relu(r_tar.abs() - r_pred * torch.sign(r_tar)) * (r_mag > 0.012).float()
        l_active = ((err_charb + 1.15 * under_res) * w_active_norm).mean()

        # Output Residual RMS Energy Anchor
        r_tar_rms = torch.sqrt((r_tar ** 2 * is_active).mean(dim=[1, 2, 3]) + 1e-8)
        r_pred_rms = torch.sqrt((r_pred ** 2 * is_active).mean(dim=[1, 2, 3]) + 1e-8)
        l_energy = F.relu(1.05 * r_tar_rms - r_pred_rms).mean()

        # 2. Quiet Flat Cel-Shading Loss
        w_quiet = is_quiet * (1.0 + 2.5 * edge_halo_zone)
        l_quiet = (err_charb * w_quiet).sum() / (w_quiet.sum() * 3.0 + 1.0)

        # 3. 1-Pixel Directional Contrast & Diagonal Continuity Loss
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]

        wx = torch.maximum(w_active_norm[:, :, :, 1:], w_active_norm[:, :, :, :-1])
        wy = torch.maximum(w_active_norm[:, :, 1:, :], w_active_norm[:, :, :-1, :])
        under_gx = F.relu(dx_t.abs() - dx_p * torch.sign(dx_t)) * (dx_t.abs() > 0.02).float()
        under_gy = F.relu(dy_t.abs() - dy_p * torch.sign(dy_t)) * (dy_t.abs() > 0.02).float()

        l_grad_xy = 0.5 * (
            ((torch.sqrt((dx_p - dx_t) ** 2 + eps ** 2) + 0.8 * under_gx) * wx).mean() +
            ((torch.sqrt((dy_p - dy_t) ** 2 + eps ** 2) + 0.8 * under_gy) * wy).mean()
        )

        dd1_p = pred[:, :, 1:, 1:] - pred[:, :, :-1, :-1]
        dd1_t = target[:, :, 1:, 1:] - target[:, :, :-1, :-1]
        dd2_p = pred[:, :, 1:, :-1] - pred[:, :, :-1, 1:]
        dd2_t = target[:, :, 1:, :-1] - target[:, :, :-1, 1:]
        wd1 = torch.maximum(w_active_norm[:, :, 1:, 1:], w_active_norm[:, :, :-1, :-1])
        wd2 = torch.maximum(w_active_norm[:, :, 1:, :-1], w_active_norm[:, :, :-1, 1:])
        l_grad_diag = 0.5 * (
            (torch.sqrt((dd1_p - dd1_t) ** 2 + eps ** 2) * wd1).mean() +
            (torch.sqrt((dd2_p - dd2_t) ** 2 + eps ** 2) * wd2).mean()
        )

        # 4. 8-Neighbor Isotropic Laplacian Loss
        lap8 = self.lap8_kernel.to(pred.device, dtype=pred.dtype)
        lap_p = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode='replicate'), lap8, groups=3)
        lap_t = F.conv2d(F.pad(target, (1, 1, 1, 1), mode='replicate'), lap8, groups=3)
        l_lap = (torch.sqrt((lap_p - lap_t) ** 2 + eps ** 2) * (0.7 * w_active_norm + 0.3 * is_quiet)).mean()

        # 5. Zero-Halo Local Envelope Clamp
        bright_halo = F.relu(pred - t_max3)
        dark_bloat = F.relu(t_min3 - pred) * (1.0 - is_core)
        l_halo = ((1.8 * bright_halo + 0.8 * dark_bloat) * (1.0 + 6.0 * edge_halo_zone)).mean()

        total = 1.5 * l_active + 4.0 * l_energy + 0.40 * l_quiet + 1.5 * l_grad_xy + 1.0 * l_grad_diag + 1.5 * l_lap + 3.5 * l_halo
        return total, {
            'l_active': l_active.item(),
            'l_quiet': l_quiet.item(),
            'l_grad': (l_grad_xy + l_grad_diag).item(),
            'l_halo': l_halo.item()
        }


class AdjacentGradientLoss(nn.Module):
    """
    First-Order Adjacent Gradient Loss (No Nyquist Blind Spot).
    """
    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.charbonnier = CharbonnierLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_tar = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_tar = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = self.charbonnier(dx_pred, dx_tar)
        loss_y = self.charbonnier(dy_pred, dy_tar)
        return self.loss_weight * (loss_x + loss_y) * 0.5


class MaskedInkingLoss(nn.Module):
    """
    Legacy Masked Inking Loss kept for backward compatibility.
    """
    def __init__(self, grad_weight: float = 0.30, ink_weight: float = 0.08, threshold: float = 0.08, eps: float = 1e-6):
        super().__init__()
        self.grad_weight = grad_weight
        self.ink_weight = ink_weight
        self.threshold = threshold
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_tar = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_tar = target[:, :, 1:, :] - target[:, :, :-1, :]

        mag_x = dx_tar.abs().mean(dim=1, keepdim=True)
        mag_y = dy_tar.abs().mean(dim=1, keepdim=True)
        mask_x = (mag_x > self.threshold).float()
        mask_y = (mag_y > self.threshold).float()

        diff_x = torch.sqrt((dx_pred - dx_tar) ** 2 + self.eps ** 2)
        diff_y = torch.sqrt((dy_pred - dy_tar) ** 2 + self.eps ** 2)

        loss_gx = (diff_x * mask_x).sum() / (mask_x.sum() + self.eps)
        loss_gy = (diff_y * mask_y).sum() / (mask_y.sum() + self.eps)
        loss_grad = (loss_gx + loss_gy) * 0.5

        edge_mask = torch.zeros_like(pred)
        edge_mask[:, :, :, 1:] = torch.maximum(edge_mask[:, :, :, 1:], mask_x)
        edge_mask[:, :, 1:, :] = torch.maximum(edge_mask[:, :, 1:, :], mask_y)

        diff_ink = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        loss_profile = (diff_ink * edge_mask).sum() / (edge_mask.sum() + self.eps)

        core_mask = (target.mean(dim=1, keepdim=True) < 0.15).float() * edge_mask
        l_core = F.relu(pred - target)
        loss_core = (l_core * core_mask).sum() / (core_mask.sum() + self.eps)

        return self.grad_weight * loss_grad + self.ink_weight * (loss_profile + 0.5 * loss_core)


class LaplacianLoss(nn.Module):
    """
    Laplacian 2nd-Order Curvature Loss.
    """
    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap', lap)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None) -> torch.Tensor:
        c = pred.size(1)
        lap = self.lap.repeat(c, 1, 1, 1).to(pred.device)
        lap_pred = F.conv2d(pred, lap, padding=1, groups=c)
        lap_tar = F.conv2d(target, lap, padding=1, groups=c)
        diff = torch.sqrt((lap_pred - lap_tar) ** 2 + self.eps ** 2)
        if mask is not None:
            return self.loss_weight * (diff * mask).sum() / (mask.sum() + 1e-6)
        return self.loss_weight * diff.mean()


class DirectionalGradientLoss(AdjacentGradientLoss):
    """Backward compatibility alias for AdjacentGradientLoss."""
    pass
