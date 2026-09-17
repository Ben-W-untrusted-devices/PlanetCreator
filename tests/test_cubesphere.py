import numpy as np

from planetcreator import cubesphere as cs


def test_face_axes_right_handed():
    for n, tu, tv in cs.FACE_AXES:
        assert np.allclose(np.cross(tu, tv), n)


def test_round_trip_random_directions():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(10_000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    face, u, v = cs.dir_to_face_uv(d)
    assert np.all((u >= -1) & (u <= 1)) and np.all((v >= -1) & (v <= 1))
    assert set(np.unique(face)) == set(range(6))
    assert np.allclose(cs.face_uv_to_dir(face, u, v), d, atol=1e-12)


def test_latlon_round_trip():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-89.9, 89.9, 1000)
    lon = rng.uniform(-180, 180, 1000)
    lat2, lon2 = cs.dir_to_latlon(cs.latlon_to_dir(lat, lon))
    assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(lon, lon2, atol=1e-9)


def test_pole_and_meridian_conventions():
    lat, lon = cs.dir_to_latlon(np.array([0, 0, 1.0]))
    assert lat == 90
    lat, lon = cs.dir_to_latlon(np.array([0, 1.0, 0]))
    assert lat == 0 and lon == 90
    # +Z face centre is the north pole; +X face centre is (0, 0).
    assert cs.dir_to_face_uv(np.array([0, 0, 1.0]))[0] == 4
    assert cs.dir_to_face_uv(np.array([1.0, 0, 0]))[0] == 0


def test_pixel_grid_orientation():
    u, v = cs.face_pixel_uv(4)
    assert u[0, 0] < u[0, -1]  # u increases with column
    assert v[0, 0] > v[-1, 0]  # v decreases with row (row 0 at +v edge)
    assert np.isclose(u[0, 0], -0.75) and np.isclose(v[0, 0], 0.75)


def test_tangent_warp_area_uniformity():
    # Solid angle per pixel should vary by well under 2x across a face (plain cube: ~3.1x).
    n = 64
    u, v = cs.face_pixel_uv(n)
    eps = 1e-4
    d = cs.face_uv_to_dir(np.zeros_like(u, dtype=np.intp), u, v)
    du = cs.face_uv_to_dir(np.zeros_like(u, dtype=np.intp), u + eps, v) - d
    dv = cs.face_uv_to_dir(np.zeros_like(u, dtype=np.intp), u, v + eps) - d
    area = np.linalg.norm(np.cross(du, dv), axis=-1)
    assert area.max() / area.min() < 1.5
