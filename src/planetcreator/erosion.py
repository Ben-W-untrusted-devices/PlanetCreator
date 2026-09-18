"""Stream-power erosion on a global equirectangular grid ("fastscape" scheme).

Each step: fill + route flow, implicit fluvial incision  dh/dt = U - K A^m S  in
downstream-first order (Braun & Willett 2013), explicit hillslope diffusion.
The implicit solve is unconditionally stable, so large time steps are fine;
a few hundred steps carve dendritic valley networks into a noise base map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from . import hydro


@dataclass
class ErosionParams:
    iters: int = 150
    dt_yr: float = 5e4
    k_fluvial: float = 1.0e-6  # for m=0.5, n=1 with A in m^2
    m: float = 0.5
    diffusion_m2_yr: float = 0.5
    sea_level: float = 0.0
    initial_scale: float = 0.35  # start from this fraction of the base map; uplift builds the rest
    uplift_mm_yr_max: float = 1.2


@njit(cache=True)
def _spl_step(h, rec, order, acc_m2, k, m, dt, dx, sea_level, W):
    """Implicit stream-power incision; ``h`` is flat and updated in place."""
    for idx in range(order.shape[0] - 1, -1, -1):
        c = order[idx]
        r = rec[c]
        if r == c or h[c] <= sea_level:
            continue
        di = abs(c // W - r // W)
        dj = abs(c % W - r % W)
        dj = min(dj, 1)  # wrapped across the seam
        dist = dx * (1.4142135 if di + dj == 2 else 1.0)
        f = dt * k * acc_m2[c] ** m / dist
        hr = max(sea_level, h[r])
        h[c] = (h[c] + f * hr) / (1.0 + f)


@njit(cache=True)
def _diffuse(h, coef, sea_level):
    H, W = h.shape
    out = h.copy()
    for i in range(1, H - 1):
        for j in range(W):
            if h[i, j] <= sea_level:
                continue
            jl = (j - 1) % W
            jr = (j + 1) % W
            lap = h[i - 1, j] + h[i + 1, j] + h[i, jl] + h[i, jr] - 4.0 * h[i, j]
            out[i, j] = h[i, j] + coef * lap
    return out


def erode(base: np.ndarray, uplift_mm_yr: np.ndarray, params: ErosionParams, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Run the simulation. Returns (height_m, flow_accumulation_km2) at the end.

    ``base`` is the noise heightmap (m); ``uplift_mm_yr`` a field of the same shape,
    zero over ocean. Ocean cells are fixed base level.
    """
    H, W = base.shape
    p = params
    dx = np.pi * hydro.EARTH_RADIUS_KM * 1000.0 / H  # metres per row
    sea = p.sea_level
    ocean = base <= sea
    h = np.where(ocean, base, sea + p.initial_scale * (base - sea)).astype(np.float64)
    h = np.ascontiguousarray(h)
    u = np.where(ocean, 0.0, uplift_mm_yr * 1e-3).astype(np.float64)  # m/yr
    area_m2 = hydro.cell_area_km2(H, W) * 1e6
    coef = p.diffusion_m2_yr * p.dt_yr / dx**2
    acc = None
    for it in range(p.iters):
        h += u * p.dt_yr
        filled = hydro.priority_flood(h, sea, 1e-3)
        rec = hydro.d8_receivers(filled, sea)
        order = hydro.flow_order(rec)
        acc = hydro.accumulate(rec, order, area_m2)
        hf = h.ravel()
        _spl_step(hf, rec, order, acc, p.k_fluvial, p.m, p.dt_yr, dx, sea, W)
        h = hf.reshape(H, W)
        if coef > 0:
            h = _diffuse(h, coef, sea)
        np.maximum(h, np.where(ocean, h, sea + 0.5), out=h)  # land never sinks below sea level
        if progress and (it % 10 == 0 or it == p.iters - 1):
            progress(it, p.iters)
    return h.astype(np.float32), (acc.reshape(H, W) / 1e6).astype(np.float32)
