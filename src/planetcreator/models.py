"""Networks: a context-conditioned UNet for terrain (height + RGB) and a PatchGAN critic."""

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


class ContextEncoder(nn.Module):
    """Encodes the wide, coarse context window. Output stride 4 over context pixels."""

    def __init__(self, cin: int, base: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 3, padding=1), _gn(base), nn.SiLU(),
            nn.Conv2d(base, 2 * base, 4, stride=2, padding=1), _gn(2 * base), nn.SiLU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1), _gn(2 * base), nn.SiLU(),
            nn.Conv2d(2 * base, 2 * base, 4, stride=2, padding=1), _gn(2 * base), nn.SiLU(),
        )
        self.out_ch = 2 * base
        self.stride = 4

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


class ContextUNet(nn.Module):
    """UNet over the patch whose bottleneck also sees (a) the context features covering the
    patch's own footprint and (b) a global pool of the whole context window."""

    def __init__(self, in_ch: int, out_ch: int, ctx_ch: int, base: int = 32, depth: int = 4, ctx_base: int = 32):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chans[:-1]:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.ctx_enc = ContextEncoder(ctx_ch, ctx_base)
        self.mid = ConvBlock(c + 2 * self.ctx_enc.out_ch, chans[-1])
        self.dec = nn.ModuleList()
        c = chans[-1]
        for ch in reversed(chans[:-1]):
            self.dec.append(ConvBlock(c + ch, ch))
            c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_pad: int, ctx_factor: int) -> torch.Tensor:
        patch = x.shape[-1]
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        feat = self.ctx_enc(ctx)
        s = ctx_factor * self.ctx_enc.stride  # fine pixels per context-feature pixel
        c0, cs = ctx_pad // s, max(patch // s, 1)
        local = F.interpolate(feat[..., c0 : c0 + cs, c0 : c0 + cs], size=x.shape[-2:], mode="bilinear", align_corners=False)
        glob = feat.mean(dim=(2, 3), keepdim=True).expand(-1, -1, *x.shape[-2:])
        x = self.mid(torch.cat([x, local, glob], dim=1))
        for blk in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN (pix2pix style): scores (cond, output) pairs on ~70 px receptive fields.

    InstanceNorm rather than spectral norm: with unit-scale inputs, spectral
    norm capped the logits so hard the discriminator barely moved off zero.
    """

    def __init__(self, in_ch: int, base: int = 32, layers: int = 3):
        super().__init__()
        seq: list[nn.Module] = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2)]
        c = base
        for i in range(1, layers):
            n = min(base * 2**i, 256)
            seq += [nn.Conv2d(c, n, 4, 2, 1), nn.InstanceNorm2d(n, affine=True), nn.LeakyReLU(0.2)]
            c = n
        seq += [nn.Conv2d(c, c, 4, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, cond: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, out], dim=1))


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
