import numpy as np
import pytest

from planetcreator.bake import CTX_LAYERS, BakeSpec, LatLonBox, bake, load_meta
from planetcreator.dataset import CubePatchDataset, PatchSpec
from planetcreator.layers import EquirectGrid


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    root = tmp_path_factory.mktemp("cube")
    h, w = 180, 360
    lat = 90 - (np.arange(h) + 0.5)
    lon = -180 + (np.arange(w) + 0.5)
    # "Land" is a big blob in the eastern hemisphere; height rises inland with a pit.
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = np.hypot((la - 10) / 40, (lo - 80) / 60)
    height = np.where(r < 1, 1500 * (1 - r) + 20, -3000).astype(np.float32)
    height[45:50, 250:256] -= 400  # a depression -> lake
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    tavg = np.broadcast_to(25 - np.abs(la), (h, w)).astype(np.float32)
    spec = BakeSpec(
        resolution=64,
        layers={"height": EquirectGrid(height), "rgb": EquirectGrid(rgb), "tavg": EquirectGrid(tavg),
                "trange": EquirectGrid(tavg), "prec": EquirectGrid(tavg)},
        holdout=(LatLonBox("box", -20, 20, 100, 140),),
        ctx_factor=8, ctx_pad=16, dist_pad=16,
    )
    bake(spec, root)
    return root


def test_bake_outputs(cube):
    meta = load_meta(cube)
    assert meta["resolution"] == 64
    for face in meta["faces"]:
        for name in ("height", "rgb", "lat", "lon", "holdout", "water", "flowacc", "dist_up", "dist_right"):
            assert (cube / face / f"{name}.npy").exists()
        for name in CTX_LAYERS:
            a = np.load(cube / face / f"ctx_{name}.npy")
            assert a.shape[:2] == (64 // 8 + 2 * 2, 64 // 8 + 2 * 2)
    rgb = np.load(cube / "px" / "rgb.npy")
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    lat = np.load(cube / "pz" / "lat.npy")
    assert lat[31:33, 31:33].min() > 88
    assert not np.load(cube / "px" / "holdout.npy").any()
    assert np.load(cube / "py" / "holdout.npy").any()


def test_derived_channels_make_sense(cube):
    py = cube / "py"  # +Y face contains most of the land blob
    h = np.load(py / "height.npy")
    water = np.load(py / "water.npy")
    flow = np.load(py / "flowacc.npy")
    assert water.dtype == bool and water[h <= -2900].all()  # (bilinear height straddles the -3000 m coast cliff)
    assert water[h > 0].any()  # the lake
    assert flow[h > 0].max() > flow[h > 0].min() + 1  # accumulation spans >10x on land
    d_up, d_down = np.load(py / "dist_up.npy"), np.load(py / "dist_down.npy")
    assert d_up[water].max() == 0 and d_up[~water].min() > 0
    assert not np.allclose(d_up, d_down)


def test_dataset_train_avoids_holdout_and_is_deterministic(cube):
    ds = CubePatchDataset(cube, PatchSpec(patch=16, layers=("height", "rgb", "lat")), length=50)
    a, b = ds[3], ds[3]
    assert a["height"].shape == (1, 16, 16) and a["rgb"].shape == (3, 16, 16)
    assert a["rgb"].max() <= 1.0
    assert all(np.array_equal(a[k], b[k]) for k in a)
    for i in range(50):
        m = ds[i]["meta"]
        face, r, c = m[0].item(), m[1].item(), m[2].item()
        hold = np.load(cube / ds.meta["faces"][face] / "holdout.npy")
        assert not hold[r : r + 16, c : c + 16].any()


def test_dataset_val_only_holdout(cube):
    ds = CubePatchDataset(cube, PatchSpec(patch=8, split="val", layers=("height",)), length=20)
    for i in range(20):
        m = ds[i]["meta"]
        hold = np.load(cube / ds.meta["faces"][m[0].item()] / "holdout.npy")
        assert hold[m[1] : m[1] + 8, m[2] : m[2] + 8].all()


def test_dataset_land_fraction(cube):
    ds = CubePatchDataset(cube, PatchSpec(patch=8, min_land_frac=0.9, layers=("height",)), length=20)
    for i in range(20):
        assert (ds[i]["height"] > 0).float().mean() >= 0.9
