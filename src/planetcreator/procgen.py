"""Procedural macro layer: a planet's height and climate fields on the cube-sphere.

This is the phase-1 stand-in for the tectonic/climate simulation: enough
structure (continents, shelves, mountain belts, latitude climate) to feed the
learned colouriser and see whether it generalises to invented worlds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cubesphere import dir_to_latlon, face_pixel_uv, face_uv_to_dir
from .noise import Perlin3, fbm, ridged, warp


@dataclass
class PlanetParams:
    seed: int = 0
    land_fraction: float = 0.3
    continent_scale: float = 1.2  # noise frequency of continents (lower = bigger continents)
    mountain_height_m: float = 6000.0
    land_height_m: float = 1000.0  # typical lowland relief
    shelf_depth_m: float = -150.0
    ocean_depth_m: float = -4200.0
    equator_temp_c: float = 28.0
    pole_temp_c: float = -18.0  # sea-level polar annual mean (Arctic-like)
    humidity: float = 1.0  # global precipitation multiplier
    lapse_rate_c_per_km: float = 6.5


class Planet:
    """Evaluates the fields for any set of unit directions ``d`` (..., 3)."""

    def __init__(self, params: PlanetParams):
        self.p = params
        s = params.seed
        self.n_cont = Perlin3(s)
        self.n_warp = Perlin3(s + 1)
        self.n_mount = Perlin3(s + 2)
        self.n_belt = Perlin3(s + 3)
        self.n_detail = Perlin3(s + 4)
        self.n_prec = Perlin3(s + 5)
        self.n_basin = Perlin3(s + 6)
        self._threshold = self._calibrate_threshold()

    # -- fields ---------------------------------------------------------------------------

    def continent_field(self, d: np.ndarray) -> np.ndarray:
        q = warp(self.n_warp, d, 0.8, 0.35)
        return fbm(self.n_cont, q, self.p.continent_scale, 5, gain=0.55)

    def _calibrate_threshold(self) -> float:
        """Continent-field quantile that yields the requested land fraction."""
        rng = np.random.default_rng(self.p.seed)
        d = rng.normal(size=(200_000, 3)).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = self.continent_field(d)
        return float(np.quantile(c, 1.0 - self.p.land_fraction))

    def height(self, d: np.ndarray) -> np.ndarray:
        """Metres relative to sea level."""
        p = self.p
        c = self.continent_field(d) - self._threshold  # >0 land
        # Land: gentle lowlands rising inland, plus mountain belts where a second field is high.
        inland = np.clip(c / 0.2, 0, 1)
        # Continental-scale basins: a slow field tilts the lowlands so interiors drain through
        # a few trunk valleys instead of radially off every coast.
        basin = 0.5 + 0.5 * fbm(self.n_basin, warp(self.n_warp, d, 0.6, 0.3), 0.9, 3)
        lowland = p.land_height_m * (0.1 + 0.9 * inland) * (0.25 + 0.75 * basin) * (0.5 + 0.5 * fbm(self.n_detail, d, 6.0, 5))
        belt = np.clip((fbm(self.n_belt, d, 1.6, 3) + 0.02) / 0.3, 0, 1) * np.clip(c / 0.08, 0, 1)
        mountains = p.mountain_height_m * belt * ridged(self.n_mount, d, 5.0, 6) ** 1.3
        land = lowland + mountains
        # Ocean: shelf near the coast, then abyssal plain with mild relief.
        depth_t = np.clip(-c / 0.12, 0, 1)
        ocean = p.shelf_depth_m + (p.ocean_depth_m - p.shelf_depth_m) * depth_t ** 0.7
        ocean += 300.0 * fbm(self.n_detail, d, 3.0, 4) * depth_t
        return np.where(c > 0, land, ocean).astype(np.float32)

    def climate(self, d: np.ndarray, height: np.ndarray, continentality: np.ndarray) -> dict[str, np.ndarray]:
        """tavg/trange (degC) and prec (mm/yr) from latitude, altitude and distance from ocean."""
        p = self.p
        lat, _ = dir_to_latlon(d)
        a = np.abs(lat).astype(np.float32) / 90.0
        # Sea-level temperature follows cos-ish falloff; continent interiors are more extreme.
        tsea = p.pole_temp_c + (p.equator_temp_c - p.pole_temp_c) * np.cos(a * np.pi / 2) ** 1.3
        tavg = tsea - p.lapse_rate_c_per_km * np.clip(height, 0, None) / 1000.0
        trange = 3.0 + 45.0 * continentality * a**0.8
        # Precipitation: ITCZ peak, subtropical dry belts, temperate secondary peak, polar dry.
        zonal = (
            1.0 * np.exp(-((a - 0.0) / 0.14) ** 2)
            + 0.6 * np.exp(-((a - 0.55) / 0.16) ** 2)
            + 0.18
            - 0.22 * np.exp(-((a - 0.27) / 0.08) ** 2)
        )
        moisture = 1.0 - 0.6 * continentality
        variation = np.exp(1.1 * fbm(self.n_prec, d, 3.0, 4))  # log-normal-ish spread like real rainfall
        prec = 3200.0 * p.humidity * np.clip(zonal, 0.02, None) * moisture * variation
        return {
            "tavg": tavg.astype(np.float32),
            "trange": trange.astype(np.float32),
            "prec": np.clip(prec, 0, None).astype(np.float32),
        }


def continentality_from_height(height: np.ndarray, radius_px: int) -> np.ndarray:
    """0 at the coast/ocean rising to 1 deep inland: a box-blurred land mask, clipped.

    Cheap stand-in for distance-to-ocean; blur radius is in pixels of the face.
    """
    land = (height > 0).astype(np.float32)
    k = 2 * radius_px + 1
    c = np.cumsum(np.pad(land, ((radius_px + 1, radius_px), (0, 0)), mode="edge"), axis=0)
    box = (c[k:] - c[:-k]) / k
    c = np.cumsum(np.pad(box, ((0, 0), (radius_px + 1, radius_px)), mode="edge"), axis=1)
    box = (c[:, k:] - c[:, :-k]) / k
    return (np.clip((box - 0.5) / 0.5, 0, 1) * land).astype(np.float32)


def generate_face(planet: Planet, face: int, n: int, coast_radius_px: int | None = None) -> dict[str, np.ndarray]:
    """All layers the colouriser needs (minus rgb) for one face at n x n."""
    u, v = face_pixel_uv(n)
    d = face_uv_to_dir(np.full_like(u, face, dtype=np.intp), u, v).astype(np.float32)
    height = planet.height(d)
    lat, lon = dir_to_latlon(d)
    cont = continentality_from_height(height, coast_radius_px or max(4, n // 40))
    layers = {"height": height, "lat": lat.astype(np.float32), "lon": lon.astype(np.float32)}
    layers.update(planet.climate(d, height, cont))
    return layers


def equirect_dirs(H: int, W: int) -> np.ndarray:
    """Unit directions at the centres of an H x W canonical equirect grid, shape (H, W, 3)."""
    from .cubesphere import latlon_to_dir

    lat = 90 - (np.arange(H) + 0.5) * 180 / H
    lon = -180 + (np.arange(W) + 0.5) * 360 / W
    return latlon_to_dir(lat[:, None], lon[None, :]).astype(np.float32)


def generate_equirect(planet: Planet, H: int, W: int, rows_per_chunk: int = 256) -> dict[str, np.ndarray]:
    """Base height and climate on an equirect grid (climate uses continentality of the base map)."""
    d = equirect_dirs(H, W)
    height = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, rows_per_chunk):
        height[r0 : r0 + rows_per_chunk] = planet.height(d[r0 : r0 + rows_per_chunk])
    return {"height": height, "dirs": d}


def uplift_field(planet: Planet, d: np.ndarray, base_height: np.ndarray, max_mm_yr: float) -> np.ndarray:
    """Uplift rate proportional to the base map's relief above the lowlands: mountain belts
    keep rising while rivers cut them; plains get a trickle so they stay above the sea."""
    p = planet.p
    land = base_height > 0
    rel = np.clip((base_height - 0.6 * p.land_height_m) / p.mountain_height_m, 0, 1)
    return np.where(land, max_mm_yr * (0.04 + 0.96 * rel**1.2), 0.0).astype(np.float64)


def climate_equirect(planet: Planet, d: np.ndarray, height: np.ndarray, rows_per_chunk: int = 256) -> dict[str, np.ndarray]:
    H, W = height.shape
    cont = continentality_from_height(height, max(4, H // 20))
    out = {k: np.empty((H, W), dtype=np.float32) for k in ("tavg", "trange", "prec")}
    for r0 in range(0, H, rows_per_chunk):
        sl = slice(r0, r0 + rows_per_chunk)
        c = planet.climate(d[sl], height[sl], cont[sl])
        for k, arr in out.items():
            arr[sl] = c[k]
    return out
