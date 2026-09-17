"""Registry of the public datasets the pipeline downloads.

Every entry is a single file (or archive) at a fixed URL. Sizes are the
Content-Length observed when the URL was verified and are only used for
progress reporting.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    size_bytes: int | None
    description: str
    license: str

    @property
    def subdir(self) -> str:
        return self.name.split("/")[0]


ETOPO_60S = Source(
    name="etopo/60s_surface",
    url=(
        "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
        "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
    ),
    filename="ETOPO_2022_v1_60s_N90W180_surface.nc",
    size_bytes=None,
    description="ETOPO 2022 global relief (ice surface + bathymetry), 60 arc-second (~1.85 km)",
    license="Public domain (NOAA NCEI)",
)


def bmng_month(yyyymm: str) -> Source:
    """NASA Blue Marble Next Generation, flat base map (no relief shading), 2 km GeoTIFF."""
    if not (len(yyyymm) == 6 and yyyymm.startswith("2004")):
        raise ValueError(f"BMNG months are 200401..200412, got {yyyymm!r}")
    month = int(yyyymm[4:])
    month_name = calendar.month_name[month].lower()
    fname = f"world.{yyyymm}.3x21600x10800_geo.tif"
    return Source(
        name=f"bmng/{yyyymm}",
        url=(
            "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/"
            f"bmng-base/{month_name}/{fname}"
        ),
        filename=fname,
        size_bytes=176_000_000,
        description=f"Blue Marble Next Generation {calendar.month_name[month]} 2004, 21600x10800 RGB",
        license="Public domain (NASA)",
    )


WORLDCLIM_TAVG = Source(
    name="worldclim/tavg",
    url="https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_tavg.zip",
    filename="wc2.1_2.5m_tavg.zip",
    size_bytes=443_224_846,
    description="WorldClim 2.1 monthly mean temperature (degC), 2.5 arc-minute",
    license="CC BY-SA 4.0",
)

WORLDCLIM_PREC = Source(
    name="worldclim/prec",
    url="https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_prec.zip",
    filename="wc2.1_2.5m_prec.zip",
    size_bytes=71_780_834,
    description="WorldClim 2.1 monthly precipitation (mm), 2.5 arc-minute",
    license="CC BY-SA 4.0",
)

DEFAULT_MONTHS = ("200407",)


def default_sources(months: tuple[str, ...] = DEFAULT_MONTHS) -> list[Source]:
    return [ETOPO_60S, WORLDCLIM_TAVG, WORLDCLIM_PREC, *(bmng_month(m) for m in months)]
