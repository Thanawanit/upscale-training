import torch
import torch.nn as nn
import torch.nn.functional as F


class LineFlowBlock(nn.Module):
    """
    Ultra-fast residual block using separable directional depthwise strips (1xK -> Kx1)
    and dilated 3x3 mix without heavy Sigmoid/exp overhead per block.
    Provides (K*dilation)x(K*dilation) 2D receptive field at minimal CPU/GPU/ONNX latency.
    """
    def __init__(self, dim: int, strip_k: int = 9, dilation: int = 1, res_scale: float = 0.25):
        super().__init__()
        self.res_scale = res_scale
        pad_s = ((strip_k - 1) * dilation) // 2

        self.conv_in = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act1 = nn.PReLU(num_parameters=dim, init=0.2)

        # Separable directional strip context (full 2D span with zero torch.cat copy overhead)
        self.dw_h = nn.Conv2d(dim, dim, (1, strip_k), 1, (0, pad_s), dilation=(1, dilation), groups=dim)
        self.dw_v = nn.Conv2d(dim, dim, (strip_k, 1), 1, (pad_s, 0), dilation=(dilation, 1), groups=dim)

        self.conv_mix = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act2 = nn.PReLU(num_parameters=dim, init=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.conv_in(x))
        # Parallel residual add of horizontal & vertical long-range context + local identity
        h_ctx = h + 0.5 * (self.dw_h(h) + self.dw_v(h))
        out = self.conv_mix(self.act2(h_ctx))
        return x + self.res_scale * out


class XyetherLineFlowNet(nn.Module):
    """
    Phase 23+ Original Fast Line-Flow Super-Resolution Architecture (100% Own Design).
    Combines hierarchical skip fusion, alternating standard/dilated directional strips
    (spanning >150 HR pixels for straight 1px strokes), a lightweight output spatial gate,
    and Nearest-Residual bounds.
    """
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 56, num_blocks: int = 8, upscale: int = 2):
        super().__init__()
        self.upscale = upscale
        self.num_feat = num_feat
        self.num_blocks = num_blocks

        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat, init=0.2),
        )

        self.blocks = nn.ModuleList([
            LineFlowBlock(
                dim=num_feat,
                strip_k=9,
                dilation=(2 if (i % 2 == 1) else 1),
                res_scale=0.25,
            )
            for i in range(num_blocks)
        ])

        tap_indices = list(range(1, num_blocks, 2))
        if (num_blocks - 1) not in tap_indices:
            tap_indices.append(num_blocks - 1)
        self.tap_set = set(tap_indices)

        self.fusion = nn.Sequential(
            nn.Conv2d(num_feat * len(tap_indices), num_feat, 1, 1, 0),
            nn.PReLU(num_parameters=num_feat, init=0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )
        # Single fast spatial-channel gate before PixelShuffle tail
        self.out_gate = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        f0 = self.head(x)
        h = f0
        taps = []
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i in self.tap_set:
                taps.append(h)
        fused = self.fusion(torch.cat(taps, dim=1))
        gated = fused * self.out_gate(fused) + f0
        residual = self.tail(gated)
        return torch.clamp(base + residual, 0.0, 1.0)
