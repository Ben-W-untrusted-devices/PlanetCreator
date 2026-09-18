import numpy as np

from planetcreator.noise import Perlin3, fbm
from planetcreator.procgen import Planet, PlanetParams, continentality_from_height, generate_face


def test_perlin_range_and_determinism():
    n = Perlin3(3)
    rng = np.random.default_rng(0)
    x, y, z = (rng.uniform(-10, 10, 5000).astype(np.float32) for _ in range(3))
    a, b = n(x, y, z), Perlin3(3)(x, y, z)
    assert np.array_equal(a, b) and a.min() > -1.05 and a.max() < 1.05 and a.std() > 0.05


def test_fbm_is_seamless_across_faces():
    # Same direction from two faces' edges gives the same value.
    from planetcreator.cubesphere import face_uv_to_dir

    n = Perlin3(1)
    d1 = face_uv_to_dir(np.array([0]), np.array([1.0]), np.array([0.3]))  # +X face, u=+1 edge
    d2 = face_uv_to_dir(np.array([2]), np.array([-1.0]), np.array([0.3]))  # +Y face, u=-1 edge
    assert np.allclose(d1, d2)
    assert np.allclose(fbm(n, d1.astype(np.float32), 2.0, 4), fbm(n, d2.astype(np.float32), 2.0, 4))


def test_land_fraction_calibration():
    planet = Planet(PlanetParams(seed=7, land_fraction=0.35))
    rng = np.random.default_rng(1)
    d = rng.normal(size=(50_000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    frac = (planet.height(d) > 0).mean()
    assert abs(frac - 0.35) < 0.03


def test_generate_face_layers_sane():
    layers = generate_face(Planet(PlanetParams(seed=2)), 4, 64)
    assert set(layers) == {"height", "lat", "lon", "tavg", "trange", "prec"}
    assert all(v.shape == (64, 64) and v.dtype == np.float32 for v in layers.values())
    assert layers["lat"].max() > 85  # +Z face holds the pole
    assert layers["tavg"].max() < 30 and layers["tavg"].min() > -70
    assert layers["prec"].min() >= 0 and layers["prec"].max() < 6000
    assert layers["height"].max() < 9000 and layers["height"].min() > -6000


def test_continentality():
    h = np.full((40, 40), -1000.0, dtype=np.float32)
    h[10:30, 10:30] = 500.0
    c = continentality_from_height(h, 4)
    assert c[20, 20] == 1.0 and c[10, 10] < 0.3 and c[0, 0] == 0.0
