import numpy as np

from planetcreator.layers import EquirectGrid
from planetcreator.reproject import sample_equirect


def _field(lat, lon):
    return np.sin(np.radians(lat)) * 3 + np.cos(np.radians(lon)) * 2


def _grid(h=180, w=360, channels=None):
    lat = 90 - (np.arange(h) + 0.5) * (180 / h)
    lon = -180 + (np.arange(w) + 0.5) * (360 / w)
    f = _field(lat[:, None], lon[None, :]).astype(np.float32)
    if channels:
        f = np.stack([f + c for c in range(channels)], axis=-1)
    return EquirectGrid(f)


def test_bilinear_matches_smooth_field():
    g = _grid()
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-85, 85, 500), rng.uniform(-180, 180, 500)
    out = sample_equirect(g, lat, lon)
    assert np.allclose(out, _field(lat, lon), atol=2e-3)


def test_longitude_wrap_is_continuous():
    g = _grid()
    a = sample_equirect(g, np.array([10.0]), np.array([179.999]))
    b = sample_equirect(g, np.array([10.0]), np.array([-179.999]))
    assert abs(a - b) < 1e-3


def test_multichannel_and_nearest():
    g = _grid(channels=3)
    out = sample_equirect(g, np.array([0.0, 45.0]), np.array([0.0, 90.0]))
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 1] - out[:, 0], 1, atol=1e-5)
    near = sample_equirect(g, np.array([0.0]), np.array([0.0]), nearest=True)
    assert near.shape == (1, 3) and near.dtype == np.float32


def test_uint8_input_is_promoted():
    g = EquirectGrid(np.full((10, 20, 3), 200, dtype=np.uint8))
    out = sample_equirect(g, np.array([0.0]), np.array([0.0]))
    assert out.dtype == np.float32 and np.allclose(out, 200)


def test_zonal_fill_smooths_sparse_rows():
    from planetcreator.layers import _fill_nodata_zonal

    a = np.zeros((10, 8), dtype=np.float32)
    valid = np.zeros_like(a, dtype=bool)
    a[2, 0] = 100  # one lonely valid pixel in row 2
    valid[2, 0] = True
    a[3:6, :4] = 10  # dense rows 3..5
    valid[3:6, :4] = True
    out = _fill_nodata_zonal(a, valid, band_rows=3)
    assert np.allclose(out[valid], a[valid])  # valid cells untouched
    assert out[2, 5] < 100  # sparse row is pulled toward its dense neighbours
    assert np.allclose(out[0], out[1])  # rows beyond the data take one wide-band value
    assert np.allclose(out[9], out[6]) and 10 < out[9, 0] < 100  # wide-band mean of all data


def test_worldclim_loader_drops_all_zero_temperature(tmp_path):
    import zipfile

    import rasterio
    from rasterio.transform import from_origin

    from planetcreator.layers import load_worldclim

    h, w = 6, 12
    zpath = tmp_path / "wc.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for m in range(1, 13):
            a = np.full((h, w), -3.4e38, dtype=np.float32)
            a[2:5, 2:8] = 10 + m  # real land
            a[2, 2] = 0.0  # bogus pixel: exactly zero in every month
            a[4, 7] = 0.0 if m == 6 else 5.0  # legit: crosses zero in one month
            tif = tmp_path / f"wc2.1_2.5m_tavg_{m:02d}.tif"
            with rasterio.open(
                tif, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                nodata=-3.4e38, transform=from_origin(-180, 90, 30, 30), crs="EPSG:4326",
            ) as dst:
                dst.write(a, 1)
            zf.write(tif, tif.name)
    out = load_worldclim(zpath, "tavg")
    land = out["land"].data
    assert not land[2, 2] and land[4, 7] and land[3, 3]
    assert np.isfinite(out["tavg"].data).all()
    assert out["tavg"].data[2, 2] != 0.0  # filled from the band mean, not left at zero
