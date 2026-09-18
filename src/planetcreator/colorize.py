"""Colourisation model: conditioning layers -> satellite RGB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import COND_CHANNELS, TRAIN_PX_KM, build_cond
from .models import PatchDiscriminator, UNet, hinge_d_loss, hinge_g_loss


@dataclass
class ColorizeConfig:
    base: int = 32
    depth: int = 4
    grad_weight: float = 0.5  # weight of the image-gradient L1 term
    adv_weight: float = 0.0  # weight of the adversarial term; 0 disables the discriminator


class Colorizer(nn.Module):
    def __init__(self, cfg: ColorizeConfig | None = None):
        super().__init__()
        self.cfg = cfg or ColorizeConfig()
        self.net = UNet(len(COND_CHANNELS), 3, self.cfg.base, self.cfg.depth)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(cond))

    def make_discriminator(self) -> PatchDiscriminator:
        return PatchDiscriminator(len(COND_CHANNELS) + 3)

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fake_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Generator loss: L1 + gradient L1 (+ hinge adversarial when logits are given)."""
        l1 = F.l1_loss(pred, target)
        gp = pred[..., 1:, :] - pred[..., :-1, :], pred[..., :, 1:] - pred[..., :, :-1]
        gt = target[..., 1:, :] - target[..., :-1, :], target[..., :, 1:] - target[..., :, :-1]
        grad = F.l1_loss(gp[0], gt[0]) + F.l1_loss(gp[1], gt[1])
        total = l1 + self.cfg.grad_weight * grad
        parts = {"l1": l1.item(), "grad": grad.item()}
        if fake_logits is not None and self.cfg.adv_weight > 0:
            adv = hinge_g_loss(fake_logits)
            total = total + self.cfg.adv_weight * adv
            parts["adv"] = adv.item()
        return total, parts

    @staticmethod
    def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return hinge_d_loss(real_logits, fake_logits)

    def save(self, path: Path, **extra) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path, map_location="cpu") -> tuple[Colorizer, dict]:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(ColorizeConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m, ck


def _hann2d(n: int, device) -> torch.Tensor:
    w = torch.hann_window(n, periodic=False, device=device).clamp(min=1e-3)
    return w[:, None] * w[None, :]


@torch.no_grad()
def tile_infer(
    model: Colorizer,
    layers: dict[str, np.ndarray],
    tile: int = 256,
    overlap: int = 64,
    batch: int = 8,
    device: str = "cpu",
    sea_level: float = 0.0,
    px_km: float = TRAIN_PX_KM,
) -> np.ndarray:
    """Run the model over full (N, N) layer arrays with overlapping, Hann-blended tiles.

    Returns uint8 (N, N, 3). Tiles that would run past the edge are shifted back
    inside, so every pixel is covered.
    """
    model.eval().to(device)
    n = layers["height"].shape[0]
    stride = tile - overlap
    starts = sorted({min(s, n - tile) for s in range(0, n, stride)})
    acc = torch.zeros(3, n, n, device=device)
    wsum = torch.zeros(1, n, n, device=device)
    win = _hann2d(tile, device)
    coords = [(i, j) for i in starts for j in starts]
    for k in range(0, len(coords), batch):
        chunk = coords[k : k + batch]
        b = {
            name: torch.stack(
                [torch.from_numpy(np.ascontiguousarray(a[i : i + tile, j : j + tile], dtype=np.float32)) for i, j in chunk]
            )[:, None].to(device)
            for name, a in layers.items()
        }
        pred = model(build_cond(b, sea_level, px_km))
        for (i, j), p in zip(chunk, pred):
            acc[:, i : i + tile, j : j + tile] += p * win
            wsum[:, i : i + tile, j : j + tile] += win
    out = (acc / wsum).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (out * 255 + 0.5).astype(np.uint8)


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
