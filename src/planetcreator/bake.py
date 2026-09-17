"""Reproject equirectangular layers onto cube-sphere faces and write them to disk.

Output layout (``root`` = e.g. ``data/cube/4096``)::

    root/meta.json
    root/<face>/<layer>.npy      face in px, nx, py, ny, pz, nz

Every layer is an ``(N, N)`` or ``(N, N, C)`` array, memmap-friendly.
Also written per face: ``lat.npy``, ``lon.npy`` (float32 degrees) and
``holdout.npy`` (bool, True inside any held-out lat/lon box).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cubesphere import FACE_NAMES, face_pixel_latlon
from .layers import EquirectGrid
from .reproject import sample_equirect


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
    layers: dict[str, EquirectGrid]
    nearest: set[str] = field(default_factory=set)  # layers sampled without interpolation
    holdout: tuple[LatLonBox, ...] = DEFAULT_HOLDOUT


def bake(spec: BakeSpec, root: Path, faces: range = range(6), progress=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = spec.resolution
    for f in faces:
        fdir = root / FACE_NAMES[f]
        fdir.mkdir(exist_ok=True)
        lat, lon = face_pixel_latlon(f, n)
        np.save(fdir / "lat.npy", lat.astype(np.float32))
        np.save(fdir / "lon.npy", lon.astype(np.float32))
        hold = np.zeros((n, n), dtype=bool)
        for box in spec.holdout:
            hold |= box.contains(lat, lon)
        np.save(fdir / "holdout.npy", hold)
        for name, grid in spec.layers.items():
            if progress:
                progress(f"{FACE_NAMES[f]}/{name}")
            out = sample_equirect(grid, lat, lon, nearest=name in spec.nearest)
            if grid.data.dtype == np.uint8 and name not in spec.nearest:
                out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
            np.save(fdir / f"{name}.npy", out)
    meta = {
        "resolution": n,
        "faces": list(FACE_NAMES),
        "layers": {
            name: {"dtype": str(np.load(root / FACE_NAMES[0] / f"{name}.npy", mmap_mode="r").dtype),
                   "shape": list(np.load(root / FACE_NAMES[0] / f"{name}.npy", mmap_mode="r").shape)}
            for name in spec.layers
        },
        "holdout": [vars(b) for b in spec.holdout],
        "convention": "see planetcreator.cubesphere docstring",
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))


def load_meta(root: Path) -> dict:
    return json.loads((root / "meta.json").read_text())
