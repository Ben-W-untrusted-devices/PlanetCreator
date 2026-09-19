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
    routing_power: float = 0.0  # >0: stochastic receiver choice ~ slope**power (0 = deterministic D8)
    k_variation: float = 0.0  # log-normal sigma of a spatially correlated erodibility field
    k_variation_cells: float = 6.0  # correlation length of that field, in cells


@njit(cache=True)
def _spl_step(h, rec, order, acc_m2, k, kfield, m, dt, dx, sea_level, W, cosr):
    """Implicit stream-power incision; ``h`` is flat and updated in place. ``kfield`` scales
    the erodibility per cell (1 = uniform); ``cosr`` is the per-row E-W spacing factor."""
    for idx in range(order.shape[0] - 1, -1, -1):
        c = order[idx]
        r = rec[c]
        if r == c or h[c] <= sea_level:
            continue
        i = c // W
        di = abs(i - r // W)
        dj = min(abs(c % W - r % W), 1)  # wrapped across the seam
        dist = dx * np.sqrt(di * di + (dj * cosr[i]) ** 2)
        f = dt * k * kfield[c] * acc_m2[c] ** m / dist
        hr = max(sea_level, h[r])
        h[c] = (h[c] + f * hr) / (1.0 + f)


@njit(cache=True)
def _diffuse(h, coef, sea_level, cosr):
    H, W = h.shape
    out = h.copy()
    for i in range(1, H - 1):
        cx = 1.0 / (cosr[i] * cosr[i])  # E-W second difference over a shorter spacing
        for j in range(W):
            if h[i, j] <= sea_level:
                continue
            jl = (j - 1) % W
            jr = (j + 1) % W
            lap = (h[i - 1, j] + h[i + 1, j] - 2.0 * h[i, j]) + cx * (h[i, jl] + h[i, jr] - 2.0 * h[i, j])
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
    cosr = hydro.row_cos(H, 0.15)
    # explicit diffusion must stay stable where E-W cells are short: cap the coefficient
    coef = min(p.diffusion_m2_yr * p.dt_yr / dx**2, 0.2 * 0.15**2)
    kfield = np.ones(H * W)
    if p.k_variation > 0:
        # Lithology stand-in: smooth log-normal field so channels wander around resistant rock.
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(int(abs(base).sum()) % (2**32))
        noise = gaussian_filter(rng.normal(size=(H, W)), p.k_variation_cells, mode=("nearest", "wrap"))
        noise *= p.k_variation / max(noise.std(), 1e-9)
        kfield = np.exp(noise).ravel()
    acc = None
    for it in range(p.iters):
        h += u * p.dt_yr
        filled = hydro.priority_flood(h, sea, 1e-3)
        if p.routing_power > 0:
            rec = hydro.d8_receivers_stochastic(filled, sea, p.routing_power, it, cosr)
        else:
            rec = hydro.d8_receivers(filled, sea, cosr)
        order = hydro.flow_order(rec)
        acc = hydro.accumulate(rec, order, area_m2)
        hf = h.ravel()
        _spl_step(hf, rec, order, acc, p.k_fluvial, kfield, p.m, p.dt_yr, dx, sea, W, cosr)
        h = hf.reshape(H, W)
        if coef > 0:
            h = _diffuse(h, coef, sea, cosr)
        np.maximum(h, np.where(ocean, h, sea + 0.5), out=h)  # land never sinks below sea level
        if progress and (it % 10 == 0 or it == p.iters - 1):
            progress(it, p.iters)
    return h.astype(np.float32), (acc.reshape(H, W) / 1e6).astype(np.float32)
