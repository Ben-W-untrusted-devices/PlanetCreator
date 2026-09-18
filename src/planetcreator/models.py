"""Small image-to-image UNet used for colourisation (and later refinement)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """``depth`` downsamplings of 2x; channels double per level from ``base``."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 32, depth: int = 4):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chans[:-1]:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.mid = ConvBlock(c, chans[-1])
        self.dec = nn.ModuleList()
        c = chans[-1]
        for ch in reversed(chans[:-1]):
            self.dec.append(ConvBlock(c + ch, ch))
            c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x)
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)
