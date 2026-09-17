"""Sample a canonical equirectangular grid at arbitrary lat/lon (bilinear, lon-wrapping)."""

from __future__ import annotations

import numpy as np

from .layers import EquirectGrid


def sample_equirect(grid: EquirectGrid, lat: np.ndarray, lon: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Return values at (lat, lon) degrees; output shape ``lat.shape (+ C)``, float32 or the
    grid dtype for ``nearest``.

    Longitude wraps; latitude is clamped at the poles.
    """
    h, w = grid.shape
    # Continuous pixel coordinates of the sample points (cell centres at integer coords).
    r = (90.0 - lat) / 180.0 * h - 0.5
    c = (lon + 180.0) / 360.0 * w - 0.5

    if nearest:
        ri = np.clip(np.rint(r).astype(np.intp), 0, h - 1)
        ci = np.rint(c).astype(np.intp) % w
        return grid.data[ri, ci]

    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    fr = (r - r0).astype(np.float32)
    fc = (c - c0).astype(np.float32)
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r0 + 1, 0, h - 1)
    c0w = c0 % w
    c1w = (c0 + 1) % w

    d = grid.data
    if d.ndim == 3:
        fr, fc = fr[..., None], fc[..., None]
    top = d[r0c, c0w] * (1 - fc) + d[r0c, c1w] * fc
    bot = d[r1c, c0w] * (1 - fc) + d[r1c, c1w] * fc
    return (top * (1 - fr) + bot * fr).astype(np.float32)
