"""Terrain model: conditioning + context -> RGB (``colour`` mode) or fine height + RGB (``joint``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .bake import CTX_LAYERS
from .features import (
    COND_CHANNELS,
    CTX_CHANNELS,
    HEIGHT_SCALE,
    TRAIN_PX_KM,
    build_cond,
    build_ctx,
    coarse_from_ctx,
    noise_field,
)
from .models import ContextUNet, PatchDiscriminator, hinge_d_loss, hinge_g_loss


@dataclass
class TerrainConfig:
    mode: str = "colour"  # or "joint"
    base: int = 32
    depth: int = 4
    ctx_base: int = 32
    ctx_pad: int = 896
    ctx_factor: int = 8
    grad_weight: float = 0.5  # image-gradient L1 term (rgb and height)
    height_weight: float = 2.0  # weight of the height terms in joint mode
    adv_weight: float = 0.0  # adversarial term; 0 disables the discriminator


class TerrainNet(nn.Module):
    def __init__(self, cfg: TerrainConfig | None = None):
        super().__init__()
        self.cfg = cfg or TerrainConfig()
        out_ch = 4 if self.cfg.mode == "joint" else 3
        self.net = ContextUNet(len(COND_CHANNELS), out_ch, len(CTX_CHANNELS), self.cfg.base, self.cfg.depth, self.cfg.ctx_base)

    @property
    def out_channels(self) -> int:
        return 4 if self.cfg.mode == "joint" else 3

    def inputs(self, batch: dict[str, torch.Tensor], px_km: float = TRAIN_PX_KM) -> tuple[torch.Tensor, torch.Tensor]:
        cond = build_cond(batch, self.cfg.mode, px_km=px_km, ctx_pad=self.cfg.ctx_pad, ctx_factor=self.cfg.ctx_factor)
        return cond, build_ctx(batch)

    def coarse_nearest(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Normalised nearest-upsampled coarse height for the mean-preserving residual."""
        if self.cfg.mode != "joint":
            return None
        up = coarse_from_ctx(batch, batch["lat"].shape[-1], self.cfg.ctx_pad, self.cfg.ctx_factor)
        return up["height_nearest"] / HEIGHT_SCALE

    def predict(self, batch: dict[str, torch.Tensor], px_km: float = TRAIN_PX_KM) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Convenience: inputs + forward with the mean-preserving residual. Returns (out, cond, ctx)."""
        cond, ctx = self.inputs(batch, px_km)
        return self(cond, ctx, self.coarse_nearest(batch)), cond, ctx

    def forward(self, cond: torch.Tensor, ctx: torch.Tensor, coarse_nearest: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Returns ``rgb`` in [0, 1] and, in joint mode, ``height`` normalised (/HEIGHT_SCALE).

        The height head predicts a residual over the bilinearly upsampled coarse height
        (channel 0 of ``cond``). The residual is made **mean-preserving**: every
        ``ctx_factor``-sized block of the output averages exactly to the coarse texel
        (``coarse_nearest``, normalised), so the coarse level is always the block mean of
        the fine level and LOD transitions do not pop.
        """
        y = self.net(cond, ctx, self.cfg.ctx_pad, self.cfg.ctx_factor)
        out = {"rgb": torch.sigmoid(y[:, :3])}
        if self.cfg.mode == "joint":
            f = self.cfg.ctx_factor
            up = cond[:, :1]
            r = y[:, 3:4]
            r = r - self._block_mean(r, f)
            if coarse_nearest is not None:
                r = r + coarse_nearest - self._block_mean(up, f)
            out["height"] = up + r
        return out

    @staticmethod
    def _block_mean(x: torch.Tensor, f: int) -> torch.Tensor:
        return F.interpolate(F.avg_pool2d(x, f), scale_factor=f, mode="nearest")

    def target(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        t = {"rgb": batch["rgb"]}
        if self.cfg.mode == "joint":
            t["height"] = batch["height"] / HEIGHT_SCALE
        return t

    @staticmethod
    def _grad_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]) + F.l1_loss(
            a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
        )

    def loss(self, pred: dict, target: dict, fake_logits: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = F.l1_loss(pred["rgb"], target["rgb"])
        grad = self._grad_l1(pred["rgb"], target["rgb"])
        total = l1 + self.cfg.grad_weight * grad
        parts = {"l1": l1.item(), "grad": grad.item()}
        if "height" in pred:
            hl1 = F.l1_loss(pred["height"], target["height"])
            hgrad = self._grad_l1(pred["height"], target["height"])
            total = total + self.cfg.height_weight * (hl1 + self.cfg.grad_weight * hgrad)
            parts["h_l1_m"] = hl1.item() * HEIGHT_SCALE
        if fake_logits is not None and self.cfg.adv_weight > 0:
            adv = hinge_g_loss(fake_logits)
            total = total + self.cfg.adv_weight * adv
            parts["adv"] = adv.item()
        return total, parts

    @staticmethod
    def stack_out(out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Outputs as one tensor for the discriminator."""
        return torch.cat([out["rgb"], out["height"]], dim=1) if "height" in out else out["rgb"]

    def make_discriminator(self) -> PatchDiscriminator:
        return PatchDiscriminator(len(COND_CHANNELS) + self.out_channels)

    @staticmethod
    def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return hinge_d_loss(real_logits, fake_logits)

    def save(self, path: Path, **extra) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path, map_location="cpu") -> tuple[TerrainNet, dict]:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(TerrainConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m, ck


def _hann2d(n: int, device) -> torch.Tensor:
    w = torch.hann_window(n, periodic=False, device=device).clamp(min=1e-3)
    return w[:, None] * w[None, :]


@torch.no_grad()
def infer_face(
    model: TerrainNet,
    fine: dict[str, np.ndarray],
    ctx: dict[str, np.ndarray],
    tile: int = 256,
    overlap: int = 64,
    batch: int = 8,
    device: str = "cpu",
    px_km: float = TRAIN_PX_KM,
    face: int = 0,
    seed: int = 0,
    origin_i: int = 0,
    origin_j: int = 0,
) -> dict[str, np.ndarray]:
    """Run the model over a whole face with overlapping, Hann-blended tiles.

    ``fine`` holds the (N, N) per-pixel layers the mode needs; ``ctx`` the padded
    ``ctx_<layer>`` arrays (without the prefix). ``face``/``seed`` key the deterministic
    noise channel; ``origin_i/j`` is the fine-pixel coordinate of ``fine[0, 0]`` on the
    face (non-zero when ``fine`` is itself a padded raster). Returns ``rgb`` uint8
    (N, N, 3) and, in joint mode, ``height`` float32 metres.
    """
    model.eval().to(device)
    cfg = model.cfg
    n = fine["lat"].shape[0]
    f, pad = cfg.ctx_factor, cfg.ctx_pad
    stride = (tile - overlap) // f * f
    starts = sorted({min(s, n - tile) for s in range(0, n, stride)})
    cs = (tile + 2 * pad) // f
    oc = model.out_channels
    acc = torch.zeros(oc, n, n, device=device)
    wsum = torch.zeros(1, n, n, device=device)
    win = _hann2d(tile, device)
    coords = [(i, j) for i in starts for j in starts]
    for k in range(0, len(coords), batch):
        chunk = coords[k : k + batch]
        b: dict[str, torch.Tensor] = {}
        for name, a in fine.items():
            b[name] = torch.stack([torch.from_numpy(np.ascontiguousarray(a[i : i + tile, j : j + tile], dtype=np.float32)) for i, j in chunk])[:, None].to(device)
        for name in CTX_LAYERS:
            a = ctx[name]
            b[f"ctx_{name}"] = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(a[i // f : i // f + cs, j // f : j // f + cs], dtype=np.float32)) for i, j in chunk]
            )[:, None].to(device)
        b["noise"] = torch.stack(
            [torch.from_numpy(noise_field(face, i + origin_i, j + origin_j, tile, seed)) for i, j in chunk]
        )[:, None].to(device)
        out = model.stack_out(model.predict(b, px_km)[0])
        for (i, j), o in zip(chunk, out):
            acc[:, i : i + tile, j : j + tile] += o * win
            wsum[:, i : i + tile, j : j + tile] += win
    out = (acc / wsum).permute(1, 2, 0).cpu().numpy()
    result = {"rgb": (np.clip(out[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)}
    if oc == 4:
        result["height"] = (out[..., 3] * HEIGHT_SCALE).astype(np.float32)
    return result


def coarse_layers_for_face(ctx: dict[str, np.ndarray], n: int, pad: int, factor: int) -> dict[str, np.ndarray]:
    """Upsampled coarse height for a face from its context arrays (for previews / fallbacks)."""
    b = {f"ctx_{k}": torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))[None, None] for k, v in ctx.items()}
    up = coarse_from_ctx(b, n, pad, factor)
    return {k: v[0, 0].numpy() for k, v in up.items()}


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
