"""Load raw datasets into a canonical global equirectangular grid.

Canonical orientation: row 0 is the northern edge, column 0 the western
edge (lon -180). Cell (i, j) is centred at
``lat = 90 - (i + 0.5) * dlat`` and ``lon = -180 + (j + 0.5) * dlon``.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EquirectGrid:
    data: np.ndarray  # (H, W) or (H, W, C)
    nodata: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]


def load_etopo(path: Path) -> EquirectGrid:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        lat = ds["lat"][:]
        z = np.asarray(ds["z"][:], dtype=np.float32)
    if lat[0] < lat[-1]:  # south-up in the file; flip to north-up
        z = z[::-1]
    return EquirectGrid(np.ascontiguousarray(z))


def load_bmng(path: Path) -> EquirectGrid:
    import rasterio

    with rasterio.open(path) as src:
        rgb = src.read()  # (3, H, W) uint8
    return EquirectGrid(np.ascontiguousarray(np.moveaxis(rgb, 0, -1)))


def _fill_nodata_zonal(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid cells (oceans in WorldClim) with the row's mean of valid cells.

    A latitude-band mean is a crude sea-surface proxy but keeps the
    conditioning field smooth across coastlines.
    """
    out = a.copy()
    row_sum = np.where(valid, a, 0).sum(axis=1)
    row_cnt = valid.sum(axis=1)
    row_mean = np.where(row_cnt > 0, row_sum / np.maximum(row_cnt, 1), np.nan)
    # Rows with no land at all (open Southern Ocean): interpolate from neighbours.
    ok = ~np.isnan(row_mean)
    row_mean = np.interp(np.arange(len(row_mean)), np.flatnonzero(ok), row_mean[ok])
    out[~valid] = np.broadcast_to(row_mean[:, None], a.shape)[~valid]
    return out


def load_worldclim(zip_path: Path, var: str) -> dict[str, EquirectGrid]:
    """Derive annual summaries from the 12 monthly GeoTIFFs inside a WorldClim zip.

    Returns ``{"tavg": mean, "trange": max-min}`` for temperature or
    ``{"prec": annual total}`` for precipitation, plus ``"land"`` (valid mask).
    """
    import rasterio

    months = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".tif"))
        if len(names) != 12:
            raise ValueError(f"expected 12 monthly tifs in {zip_path}, found {len(names)}")
        for n in names:
            with zf.open(n) as fh, rasterio.MemoryFile(fh.read()) as mem, mem.open() as src:
                months.append(src.read(1).astype(np.float32))
                nodata = src.nodata
    stack = np.stack(months)
    valid = np.all(stack != nodata, axis=0) if nodata is not None else np.isfinite(stack).all(0)
    out: dict[str, EquirectGrid] = {"land": EquirectGrid(valid)}
    if var == "tavg":
        out["tavg"] = EquirectGrid(_fill_nodata_zonal(stack.mean(0), valid))
        out["trange"] = EquirectGrid(_fill_nodata_zonal(stack.max(0) - stack.min(0), valid))
    elif var == "prec":
        out["prec"] = EquirectGrid(_fill_nodata_zonal(stack.sum(0), valid))
    else:
        raise ValueError(var)
    return out
