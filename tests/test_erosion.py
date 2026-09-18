import numpy as np

from planetcreator import hydro
from planetcreator.erosion import ErosionParams, erode


def test_erosion_carves_dendritic_network():
    H, W = 96, 192
    rng = np.random.default_rng(0)
    i, j = np.mgrid[0:H, 0:W]
    # an island: a smooth dome with noise, ocean elsewhere
    r = np.hypot((i - H / 2) / (H / 3), (j - W / 2) / (W / 3))
    base = np.where(r < 1, 2500 * (1 - r) + rng.normal(0, 40, (H, W)), -3000).astype(np.float64)
    uplift = np.where(base > 0, 0.8, 0.0)
    h, acc = erode(base, uplift, ErosionParams(iters=40, dt_yr=1e5))
    land = h > 0
    assert land.sum() > 0.9 * (base > 0).sum()  # island survives
    # drainage: some cells collect much more than a single cell's area
    cell = hydro.cell_area_km2(H, W).max()
    assert acc[land].max() > 50 * cell
    # erosion produced relief variance beyond the smooth dome: valleys exist
    smooth = base.copy()
    assert np.std(np.diff(h[H // 2, :])) > 0.5 * np.std(np.diff(smooth[H // 2, :]))
    assert np.isfinite(h).all()
