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


def _fill_nodata_zonal(a: np.ndarray, valid: np.ndarray, band_rows: int | None = None) -> np.ndarray:
    """Replace invalid cells (oceans in WorldClim) with a latitude-band mean of valid cells.

    A latitude-band mean is a crude sea-surface proxy but keeps the
    conditioning field smooth across coastlines. The band is a count-weighted
    moving average of ~1 degree so rows with only a few valid pixels (the
    northernmost land) do not produce outliers; rows with no valid data in
    range (beyond the last land row) take the nearest band's value.
    """
    h = a.shape[0]
    band_rows = band_rows or max(1, round(h / 180))
    k = np.ones(band_rows)
    row_sum = np.convolve(np.where(valid, a, 0).sum(axis=1, dtype=np.float64), k, mode="same")
    row_cnt = np.convolve(valid.sum(axis=1).astype(np.float64), k, mode="same")
    ok = np.flatnonzero(row_cnt > 0)
    row_mean = np.interp(np.arange(h), ok, row_sum[ok] / row_cnt[ok])
    # Polar gaps (beyond the first/last row with data) take a wider 5-band mean so a
    # sparse edge row does not set the whole cap.
    raw_sum = np.where(valid, a, 0).sum(axis=1, dtype=np.float64)
    raw_cnt = valid.sum(axis=1).astype(np.float64)
    wide = 5 * band_rows
    first, last = np.flatnonzero(raw_cnt)[[0, -1]]
    row_mean[:first] = raw_sum[first : first + wide].sum() / max(raw_cnt[first : first + wide].sum(), 1)
    row_mean[last + 1 :] = raw_sum[last - wide + 1 : last + 1].sum() / max(raw_cnt[last - wide + 1 : last + 1].sum(), 1)
    out = a.copy()
    out[~valid] = np.broadcast_to(row_mean[:, None].astype(a.dtype), a.shape)[~valid]
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
    if var == "tavg":
        # WorldClim 2.1 has ~16k high-latitude coastal pixels that read exactly 0.0 in
        # every month; a real annual cycle never does that, so treat them as nodata.
        valid &= ~np.all(stack == 0, axis=0)
    out: dict[str, EquirectGrid] = {"land": EquirectGrid(valid)}
    stack[:, ~valid] = 0  # keep the nodata sentinel out of the reductions
    if var == "tavg":
        out["tavg"] = EquirectGrid(_fill_nodata_zonal(stack.mean(0), valid))
        out["trange"] = EquirectGrid(_fill_nodata_zonal(stack.max(0) - stack.min(0), valid))
    elif var == "prec":
        out["prec"] = EquirectGrid(_fill_nodata_zonal(stack.sum(0), valid))
    else:
        raise ValueError(var)
    return out
