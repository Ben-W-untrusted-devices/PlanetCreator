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
