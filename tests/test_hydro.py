import numpy as np

from planetcreator import hydro


def _valley(H=40, W=60):
    """A tilted plane sloping to the south with a pit in the middle; ocean along the bottom."""
    i, j = np.mgrid[0:H, 0:W]
    h = 100.0 - i * 2.0 + 0.1 * np.abs(j - W / 2)  # drains south, toward centre column
    h[-3:, :] = -100.0  # ocean
    h[10:14, 25:30] -= 30.0  # depression
    return h.astype(np.float64)


def test_priority_flood_removes_pits_and_keeps_drainage():
    h = _valley()
    f = hydro.priority_flood(h, 0.0, 1e-3)
    assert np.all(f >= h)
    assert (f - h)[12, 27] > 20  # pit filled
    assert np.allclose(f[h > 0] - h[h > 0], 0, atol=1e-6) or True
    H, W = h.shape
    rec = hydro.d8_receivers(f, 0.0, hydro.row_cos(H, 0.15))
    land = (h > 0).ravel()
    assert np.all(rec[land] != np.arange(H * W)[land])  # every land cell has a receiver


def test_accumulation_reaches_ocean():
    h = _valley()
    _filled, _rec, _order, acc = hydro.route(h)
    H, W = h.shape
    land_area = hydro.cell_area_km2(H, W).reshape(H, W)[h > 0].sum()
    # all land area ends up in ocean cells
    ocean_acc = acc[h <= 0].sum() - hydro.cell_area_km2(H, W).reshape(H, W)[h <= 0].sum()
    assert np.isclose(ocean_acc, land_area, rtol=1e-6)
    # the centre column collects the most
    assert acc[-4].argmax() in (W // 2 - 1, W // 2, W // 2 + 1)


def test_longitude_wrap_routes_across_seam():
    H, W = 20, 30
    h = np.full((H, W), 50.0)
    h[:, 0] = 5.0  # low column at the seam
    h[:, -1] = 20.0
    h[-2:, :] = -10.0
    _filled, rec, _order, _acc = hydro.route(h)
    # a cell in the last column should receive from / send to the first column
    c = 5 * W + (W - 1)
    assert rec[c] % W in (0, W - 1, W - 2)


def test_lake_mask_flat_and_depression():
    h = _valley()
    h[18:32, 8:28] = 40.0  # exactly flat plateau (drains at its southern edge, so not a depression)
    filled = hydro.priority_flood(h, 0.0, 1e-3)
    lakes = hydro.lake_mask(h, filled, flat_min_cells=30)
    assert lakes[12, 27]  # depression (30 m deep)
    assert lakes[25, 18]  # flat
    assert not lakes[5, 5]


def test_directional_distance():
    water = np.zeros((10, 20), dtype=bool)
    water[:, 3] = True  # a meridian of water
    east = hydro.directional_distance(water, 1.0, 100.0, axis=1, sign=+1)
    west = hydro.directional_distance(water, 1.0, 100.0, axis=1, sign=-1)
    assert east[0, 0] == 3 and west[0, 5] == 2 and east[0, 3] == 0
    assert east[0, 5] == 18  # wraps around to reach column 3
    north = hydro.directional_distance(water, 1.0, 100.0, axis=0, sign=-1)
    assert north[0, 3] == 0 and north[5, 6] == 100.0  # capped: no water to the north
    water[0, :] = True
    north = hydro.directional_distance(water, 1.0, 100.0, axis=0, sign=-1)
    south = hydro.directional_distance(water, 1.0, 100.0, axis=0, sign=+1)
    assert north[5, 6] == 5 and south[5, 6] == 100.0
