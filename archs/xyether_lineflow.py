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


class XyetherLineFlowChrDeblockNet(XyetherLineFlowNet):
    """
    Phase 33 Own Architecture: XyetherLineFlowNet(56ch, 8b) + 4-Scale 2D Opponent-Chroma Deblock Head.
    - Backbone predicts crisp 2x RGB (inherits 100% own LineFlow weights & razor vector lines).
    - Chroma Head operates in Opponent Space (Y, R-G, B-G) across 1/2, 1/4, 1/8, 1/16 HR
      (where 8x8 LR = 16x16 HR H.264 4:2:0 chroma macroblocks shrink to 1x1..2x2 pixels),
      predicting (dRG, dBG) ONLY while keeping Luma Y 100% untouched.
    - Pure-black ink gate `ink_keep = clamp(Y * 25, 0, 1)` guarantees line_depth == 0.00.
    """
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 56, num_blocks: int = 8, upscale: int = 2):
        super().__init__(num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=num_feat, num_blocks=num_blocks, upscale=upscale)
        self.ce1 = nn.Sequential(nn.Conv2d(3, 24, 3, 2, 1), nn.PReLU(24, init=0.2))     # 1/2 HR
        self.ce2 = nn.Sequential(nn.Conv2d(24, 32, 3, 2, 1), nn.PReLU(32, init=0.2))    # 1/4 HR
        self.ce3 = nn.Sequential(nn.Conv2d(32, 48, 3, 2, 1), nn.PReLU(48, init=0.2))    # 1/8 HR
        self.ce4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 2, 1), nn.PReLU(48, init=0.2),                         # 1/16 HR
            nn.Conv2d(48, 48, 3, 1, 1), nn.PReLU(48, init=0.2),
        )
        self.cd3 = nn.Sequential(nn.Conv2d(96, 48, 3, 1, 1), nn.PReLU(48, init=0.2))
        self.cd2 = nn.Sequential(nn.Conv2d(80, 32, 3, 1, 1), nn.PReLU(32, init=0.2))
        self.cd1 = nn.Sequential(nn.Conv2d(56, 24, 3, 1, 1), nn.PReLU(24, init=0.2))
        # Fast 1x1 Luma-guided HR refinement (avoids heavy 3x3 convs at 1080p)
        self.c_ref = nn.Sequential(
            nn.Conv2d(27, 16, 1, 1, 0), nn.PReLU(16, init=0.2),
            nn.Conv2d(16, 2, 1, 1, 0),
        )
        self.chr_alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        Y = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
        RG = out[:, 0:1] - out[:, 1:2]
        BG = out[:, 2:3] - out[:, 1:2]
        inp = torch.cat([Y, RG, BG], dim=1)

        s1 = self.ce1(inp)
        s2 = self.ce2(s1)
        s3 = self.ce3(s2)
        s4 = self.ce4(s3)
        u3 = self.cd3(torch.cat([F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        u2 = self.cd2(torch.cat([F.interpolate(u3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        u1 = self.cd1(torch.cat([F.interpolate(u2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        u0 = F.interpolate(u1, size=out.shape[-2:], mode='bilinear', align_corners=False)

        ink_keep = torch.clamp(Y * 25.0, 0.0, 1.0)
        delta_c = self.chr_alpha * ink_keep * self.c_ref(torch.cat([u0, inp], dim=1))
        RG_n = RG + delta_c[:, 0:1]
        BG_n = BG + delta_c[:, 1:2]
        G_n = Y - 0.299 * RG_n - 0.114 * BG_n
        R_n = G_n + RG_n
        B_n = G_n + BG_n
        return torch.clamp(torch.cat([R_n, G_n, B_n], dim=1), 0.0, 1.0)

