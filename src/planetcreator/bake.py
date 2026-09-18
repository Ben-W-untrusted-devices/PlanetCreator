"""Reproject equirectangular layers onto cube-sphere faces and write them to disk.

Output layout (``root`` = e.g. ``data/cube/4096``)::

    root/meta.json
    root/<face>/<layer>.npy        fine layers, (N, N) or (N, N, C)
    root/<face>/ctx_<layer>.npy    coarse context, ((N + 2 pad) / f)^2, padded beyond the face

Fine layers: whatever equirect layers are given (Earth: height, rgb, tavg, trange, prec,
land) plus ``lat``, ``lon``, ``holdout`` and, when a fine height exists, the derived
``water`` (ocean or lake), ``flowacc`` (log10 km^2) and ``dist_{up,down,left,right}``
(log1p km to *ocean* along the face axes — a moisture-source channel). Context layers are the same fields computed from the height
box-downsampled by ``ctx_factor`` — what a generator will actually have at coarse scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import hydro
from .cubesphere import FACE_NAMES, face_pixel_latlon
from .layers import EquirectGrid
from .reproject import sample_equirect

DIST_NAMES = ("dist_up", "dist_down", "dist_left", "dist_right")
CTX_LAYERS = ("height", "water", "flowacc", *DIST_NAMES, "tavg", "trange", "prec", "lat")
EARTH_QUARTER_KM = 40_075.0 / 4


@dataclass(frozen=True)
class LatLonBox:
    """Inclusive box; ``lon_min > lon_max`` wraps across the antimeridian."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        in_lat = (lat >= self.lat_min) & (lat <= self.lat_max)
        if self.lon_min <= self.lon_max:
            in_lon = (lon >= self.lon_min) & (lon <= self.lon_max)
        else:
            in_lon = (lon >= self.lon_min) | (lon <= self.lon_max)
        return in_lat & in_lon


# Whole regions, not random patches: overlapping windows leak otherwise.
DEFAULT_HOLDOUT = (
    LatLonBox("south_america", -56, 13, -82, -34),
    LatLonBox("australia_nz", -48, -10, 112, 180),
)


@dataclass
class BakeSpec:
    resolution: int
    layers: dict[str, EquirectGrid]  # fine equirect layers sampled directly onto faces
    nearest: set[str] = field(default_factory=set)  # layers sampled without interpolation
    holdout: tuple[LatLonBox, ...] = DEFAULT_HOLDOUT
    coarse_height: EquirectGrid | None = None  # context height; default: fine height box-downsampled
    ctx_factor: int = 8
    ctx_pad: int = 896  # fine pixels of context beyond each face edge (3.5 x a 256 patch)
    flow_downsample: int = 1  # box-downsample the fine height before routing (speed)
    dist_pad: int = 1024  # fine pixels of padding for the directional distances
    dist_cap_km: float = 2500.0


def box_downsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    a = a[:h, :w]
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3)).astype(a.dtype)


def derive_hydro(height: np.ndarray, sea_level: float = 0.0) -> dict[str, EquirectGrid]:
    """``water`` (ocean or lake), ``ocean`` and ``flowacc`` (log10(1 + km^2)) from an equirect height."""
    filled, _rec, _order, acc = hydro.route(height, sea_level)
    lakes = hydro.lake_mask(height, filled, sea_level)
    ocean = height <= sea_level
    return {
        "water": EquirectGrid(ocean | lakes),
        "ocean": EquirectGrid(ocean),
        "flowacc": EquirectGrid(np.log10(1.0 + acc).astype(np.float32)),
    }


def face_distances(water_padded: np.ndarray, pad: int, px_km: float, cap_km: float) -> dict[str, np.ndarray]:
    """log1p(km) to ocean along the four raster axes, from a padded face mask, cropped to the face."""
    out = {}
    for name, axis, sign in (("dist_up", 0, -1), ("dist_down", 0, 1), ("dist_left", 1, -1), ("dist_right", 1, 1)):
        d = hydro.directional_distance(water_padded, px_km, cap_km, axis, sign, wrap=False)
        d = d[pad : d.shape[0] - pad, pad : d.shape[1] - pad] if pad else d
        out[name] = np.log1p(d).astype(np.float32)
    return out


def _sample_all(layers: dict[str, EquirectGrid], nearest: set[str], lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, grid in layers.items():
        a = sample_equirect(grid, lat, lon, nearest=name in nearest)
        if grid.data.dtype == np.uint8 and name not in nearest:
            a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
        out[name] = a
    return out


def bake(spec: BakeSpec, root: Path, faces: range = range(6), progress=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n, f = spec.resolution, spec.ctx_factor
    if spec.ctx_pad % f:
        raise ValueError("ctx_pad must be a multiple of ctx_factor")
    px_km = EARTH_QUARTER_KM / n
    fine = dict(spec.layers)
    fine_height = fine.get("height")

    # Fine derived channels (need a fine height).
    fine_nearest = set(spec.nearest)
    if fine_height is not None:
        if progress:
            progress("hydro (fine)")
        h = fine_height.data if spec.flow_downsample == 1 else box_downsample(fine_height.data, spec.flow_downsample)
        fine.update(derive_hydro(h))
        fine_ocean = fine.pop("ocean")
        fine_nearest.add("water")

    # Coarse (context) layers from the coarse height.
    if progress:
        progress("hydro (coarse)")
    coarse_h = spec.coarse_height.data if spec.coarse_height is not None else box_downsample(fine_height.data, f)
    ctx = {"height": EquirectGrid(coarse_h)}
    ctx.update(derive_hydro(coarse_h))
    ctx_ocean = ctx.pop("ocean")
    for k in ("tavg", "trange", "prec"):
        if k in fine:
            ctx[k] = fine[k]
    ctx_nearest = {"water"}

    for fi in faces:
        name = FACE_NAMES[fi]
        fdir = root / name
        fdir.mkdir(exist_ok=True)
        if progress:
            progress(f"{name}: fine layers")
        lat, lon = face_pixel_latlon(fi, n)
        np.save(fdir / "lat.npy", lat.astype(np.float32))
        np.save(fdir / "lon.npy", lon.astype(np.float32))
        hold = np.zeros((n, n), dtype=bool)
        for box in spec.holdout:
            hold |= box.contains(lat, lon)
        np.save(fdir / "holdout.npy", hold)
        for k, a in _sample_all(fine, fine_nearest, lat, lon).items():
            np.save(fdir / f"{k}.npy", a)
        if fine_height is not None:
            if progress:
                progress(f"{name}: distances")
            plat, plon = face_pixel_latlon(fi, n, spec.dist_pad)
            wpad = sample_equirect(fine_ocean, plat, plon, nearest=True)
            for k, a in face_distances(wpad, spec.dist_pad, px_km, spec.dist_cap_km).items():
                np.save(fdir / f"{k}.npy", a)

        if progress:
            progress(f"{name}: context")
        nc, pc = n // f, spec.ctx_pad // f
        clat, clon = face_pixel_latlon(fi, nc, pc)
        cs = _sample_all(ctx, ctx_nearest, clat, clon)
        cs["lat"] = clat.astype(np.float32)
        cs.update(face_distances(sample_equirect(ctx_ocean, clat, clon, nearest=True), 0, px_km * f, spec.dist_cap_km))
        for k in CTX_LAYERS:
            np.save(fdir / f"ctx_{k}.npy", cs[k])

    def info(path: Path) -> dict:
        a = np.load(path, mmap_mode="r")
        return {"dtype": str(a.dtype), "shape": list(a.shape)}

    first = root / FACE_NAMES[faces[0]]
    layer_names = sorted(p.stem for p in first.glob("*.npy") if not p.stem.startswith("ctx_"))
    meta = {
        "resolution": n,
        "faces": list(FACE_NAMES),
        "layers": {k: info(first / f"{k}.npy") for k in layer_names},
        "ctx": {"factor": f, "pad": spec.ctx_pad, "layers": {k: info(first / f"ctx_{k}.npy") for k in CTX_LAYERS}},
        "holdout": [vars(b) for b in spec.holdout],
        "convention": "see planetcreator.cubesphere docstring",
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))


def load_meta(root: Path) -> dict:
    return json.loads((root / "meta.json").read_text())
