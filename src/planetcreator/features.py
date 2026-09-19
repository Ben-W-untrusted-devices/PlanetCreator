"""Turn raw layers into the normalised conditioning tensors the models see.

Kept simple and documented, because the Godot side will have to build the same
tensors from its own procedural layers at runtime.

Two modes share one channel layout:

* ``colour``: fine height and its derived channels are known; predict RGB.
* ``joint``: only the coarse (context-resolution) height and derived channels are known;
  predict fine height and RGB. The per-pixel channels come from the centre of the context
  window upsampled to patch resolution, so the network's input width is identical.

Plus the coarse context window (``ctx_*`` layers) in both modes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .bake import CTX_LAYERS, DIST_NAMES

# Channel order of the per-pixel conditioning tensor. ``noise`` is a deterministic
# per-pixel white-noise field keyed on the pixel's global position (see
# :func:`noise_field`): the model's source of detail entropy, reproducible per tile.
COND_CHANNELS = ("height", "slope", "ocean", "water", "flowacc", *DIST_NAMES, "lat", "tavg", "trange", "prec", "noise")
CTX_CHANNELS = CTX_LAYERS  # height, water, flowacc, dist x4, tavg, trange, prec, lat

HEIGHT_SCALE = 4000.0  # m
SLOPE_SCALE = 6.0  # log1p(m per training pixel)
FLOWACC_SCALE = 6.0  # log10(1 + km^2)
DIST_SCALE = 8.0  # log1p(km)
TAVG_SCALE = 30.0  # degC
TRANGE_SCALE = 40.0  # degC
PREC_LOG_SCALE = 9.0  # log1p(mm/yr); log1p(8000) ~= 9
# Slope is expressed per *training* pixel so rasters of any resolution condition the
# model identically: a 4096-px cube face spans a quarter circumference of Earth.
TRAIN_PX_KM = 40_075.0 / 4 / 4096  # ~2.45 km


def px_km_for_face(resolution: int) -> float:
    """Kilometres per pixel of an Earth-sized cube face raster at this resolution."""
    return 40_075.0 / 4 / resolution


def slope_magnitude(height: torch.Tensor) -> torch.Tensor:
    """|grad h| in metres per pixel via central differences, replicate-padded. (B,1,H,W)."""
    hp = F.pad(height, (1, 1, 1, 1), mode="replicate")
    dx = (hp[..., 1:-1, 2:] - hp[..., 1:-1, :-2]) * 0.5
    dy = (hp[..., 2:, 1:-1] - hp[..., :-2, 1:-1]) * 0.5
    return torch.sqrt(dx * dx + dy * dy)


def _norm(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "height":
        return x / HEIGHT_SCALE
    if name in ("water", "ocean"):
        return x.float()
    if name == "flowacc":
        return x / FLOWACC_SCALE
    if name.startswith("dist_"):
        return x / DIST_SCALE
    if name == "lat":
        return x / 90.0
    if name == "tavg":
        return x / TAVG_SCALE
    if name == "trange":
        return x / TRANGE_SCALE
    if name == "prec":
        return torch.log1p(x.clamp(min=0)) / PREC_LOG_SCALE
    raise KeyError(name)


def build_ctx(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, len(CTX_CHANNELS), Pc, Pc) normalised context tensor."""
    return torch.cat([_norm(name, batch[f"ctx_{name}"]) for name in CTX_CHANNELS], dim=1)


def noise_field(face: int, i0: int, j0: int, size: int, seed: int = 0) -> np.ndarray:
    """Deterministic white noise in [-1, 1] for the ``size x size`` window whose top-left
    fine pixel is (i0, j0) on ``face``: the same pixel always gets the same value, so
    overlapping tiles agree and a quadtree node refines identically every time."""
    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    ii = np.arange(i0, i0 + size, dtype=np.uint64)[:, None]
    jj = np.arange(j0, j0 + size, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        h = (ii * np.uint64(0x9E3779B97F4A7C15)) ^ (jj * np.uint64(0xBF58476D1CE4E5B9))
        salt = (np.uint64(face + 1) * np.uint64(0x94D049BB133111EB)) ^ (np.uint64(seed) * np.uint64(0x2545F4914F6CDD1D))
        h = (h ^ salt) & m
        h ^= h >> np.uint64(31)
        h = (h * np.uint64(0x7FB5D329728EA185)) & m
        h ^= h >> np.uint64(27)
    return ((h >> np.uint64(11)).astype(np.float64) / float(1 << 53) * 2.0 - 1.0).astype(np.float32)


def coarse_from_ctx(batch: dict[str, torch.Tensor], patch: int, pad: int, factor: int) -> dict[str, torch.Tensor]:
    """The centre of the context window (the patch's own footprint) upsampled to patch
    resolution, as raw-unit layers: what a generator knows about a patch at coarse scale.

    Everything is upsampled bilinearly (the water mask becomes a soft coverage ramp);
    nearest upsampling gave the model 8-px staircases to reproduce. ``height_nearest``
    is also returned for the mean-preserving residual (see ``TerrainNet.forward``)."""
    c0, cs = pad // factor, patch // factor
    out = {}
    for name in ("height", "water", "flowacc", *DIST_NAMES):
        crop = batch[f"ctx_{name}"][..., c0 : c0 + cs, c0 : c0 + cs].float()
        out[name] = F.interpolate(crop, scale_factor=factor, mode="bilinear", align_corners=False)
    crop = batch["ctx_height"][..., c0 : c0 + cs, c0 : c0 + cs].float()
    out["height_nearest"] = F.interpolate(crop, scale_factor=factor, mode="nearest")
    return out


def build_cond(
    batch: dict[str, torch.Tensor],
    mode: str = "colour",
    sea_level: float = 0.0,
    px_km: float = TRAIN_PX_KM,
    ctx_pad: int = 896,
    ctx_factor: int = 8,
) -> torch.Tensor:
    """(B, len(COND_CHANNELS), P, P) per-pixel conditioning.

    ``px_km`` is the raster's pixel spacing; slope is rescaled to metres per training pixel.
    In ``joint`` mode the height-derived channels are taken from the context (coarse) layers.
    """
    if mode == "joint":
        src = coarse_from_ctx(batch, batch["lat"].shape[-1], ctx_pad, ctx_factor)
    else:
        src = batch
    h = src["height"] - sea_level
    slope = slope_magnitude(h) * (TRAIN_PX_KM / px_km)
    chans = [
        h / HEIGHT_SCALE,
        torch.log1p(slope) / SLOPE_SCALE,
        (h < 0).float(),
        _norm("water", src["water"]),
        _norm("flowacc", src["flowacc"]),
        *(_norm(d, src[d]) for d in DIST_NAMES),
        _norm("lat", batch["lat"]),
        _norm("tavg", batch["tavg"]),
        _norm("trange", batch["trange"]),
        _norm("prec", batch["prec"]),
        batch["noise"],
    ]
    return torch.cat(chans, dim=1)
