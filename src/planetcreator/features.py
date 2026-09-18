"""Turn a batch of raw layers into the normalised conditioning tensor the models see.

Kept deliberately simple and documented, because the Godot side will have to
build the same tensor from its own procedural layers at runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Layers the colouriser needs from the cube (dataset must yield all of these).
COND_LAYERS = ("height", "lat", "tavg", "trange", "prec")

# Channel order of the conditioning tensor.
COND_CHANNELS = ("height", "slope", "ocean", "lat", "tavg", "trange", "prec")

HEIGHT_SCALE = 4000.0  # m
SLOPE_SCALE = 6.0  # log1p(m per training pixel)
# Slope is expressed per *training* pixel so rasters of any resolution condition the
# model identically: a 4096-px cube face spans a quarter circumference of Earth.
TRAIN_PX_KM = 40_075.0 / 4 / 4096  # ~2.45 km
TAVG_SCALE = 30.0  # degC
TRANGE_SCALE = 40.0  # degC
PREC_LOG_SCALE = 9.0  # log1p(mm/yr); log1p(8000) ~= 9


def slope_magnitude(height: torch.Tensor) -> torch.Tensor:
    """|grad h| in metres per pixel via central differences, replicate-padded. (B,1,H,W)."""
    hp = F.pad(height, (1, 1, 1, 1), mode="replicate")
    dx = (hp[..., 1:-1, 2:] - hp[..., 1:-1, :-2]) * 0.5
    dy = (hp[..., 2:, 1:-1] - hp[..., :-2, 1:-1]) * 0.5
    return torch.sqrt(dx * dx + dy * dy)


def px_km_for_face(resolution: int) -> float:
    """Kilometres per pixel of an Earth-sized cube face raster at this resolution."""
    return 40_075.0 / 4 / resolution


def build_cond(batch: dict[str, torch.Tensor], sea_level: float = 0.0, px_km: float = TRAIN_PX_KM) -> torch.Tensor:
    """(B, len(COND_CHANNELS), H, W) float tensor from raw layer tensors (each (B,1,H,W)).

    ``px_km`` is the raster's pixel spacing; slope is rescaled to metres per training pixel.
    """
    h = batch["height"] - sea_level
    slope = slope_magnitude(h) * (TRAIN_PX_KM / px_km)
    return torch.cat(
        [
            h / HEIGHT_SCALE,
            torch.log1p(slope) / SLOPE_SCALE,
            (h < 0).float(),
            batch["lat"] / 90.0,
            batch["tavg"] / TAVG_SCALE,
            batch["trange"] / TRANGE_SCALE,
            torch.log1p(batch["prec"].clamp(min=0)) / PREC_LOG_SCALE,
        ],
        dim=1,
    )
