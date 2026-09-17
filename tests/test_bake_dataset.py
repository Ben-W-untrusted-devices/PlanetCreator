import numpy as np
import pytest

from planetcreator.bake import BakeSpec, LatLonBox, bake, load_meta
from planetcreator.dataset import CubePatchDataset, PatchSpec
from planetcreator.layers import EquirectGrid


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    root = tmp_path_factory.mktemp("cube")
    h, w = 90, 180
    lat = 90 - (np.arange(h) + 0.5) * 2
    lon = -180 + (np.arange(w) + 0.5) * 2
    # "Land" is the eastern hemisphere; height is latitude in metres.
    height = np.broadcast_to(lat[:, None] * 10, (h, w)).astype(np.float32)
    height = np.where(lon[None, :] > 0, height, -1000).astype(np.float32)
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    spec = BakeSpec(
        resolution=64,
        layers={"height": EquirectGrid(height), "rgb": EquirectGrid(rgb)},
        holdout=(LatLonBox("box", -20, 20, 100, 140),),
    )
    bake(spec, root)
    return root


def test_bake_outputs(cube):
    meta = load_meta(cube)
    assert meta["resolution"] == 64 and set(meta["layers"]) == {"height", "rgb"}
    for face in meta["faces"]:
        for name in ("height", "rgb", "lat", "lon", "holdout"):
            assert (cube / face / f"{name}.npy").exists()
    rgb = np.load(cube / "px" / "rgb.npy")
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    # +Z face centre is the pole: lat ~ 90.
    lat = np.load(cube / "pz" / "lat.npy")
    assert lat[31:33, 31:33].min() > 88
    hold = np.load(cube / "px" / "holdout.npy")
    assert not hold.any()  # +X face spans lon -45..45, box is at 100..140
    assert np.load(cube / "py" / "holdout.npy").any()


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
