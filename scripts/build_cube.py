#!/usr/bin/env python
"""Bake raw datasets onto cube-sphere faces. Usage: build_cube.py --res 4096 --month 200407"""

import argparse
from pathlib import Path

from planetcreator.bake import BakeSpec, bake
from planetcreator.download import raw_path
from planetcreator.layers import load_bmng, load_etopo, load_worldclim
from planetcreator.sources import ETOPO_60S, WORLDCLIM_PREC, WORLDCLIM_TAVG, bmng_month


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--res", type=int, default=4096, help="pixels per face edge")
    ap.add_argument("--month", default="200407", help="BMNG month for the rgb layer")
    ap.add_argument("--out", type=Path, default=None, help="default data/cube/<res>")
    args = ap.parse_args()
    out = args.out or args.data_dir / "cube" / str(args.res)

    print("loading ETOPO"); layers = {"height": load_etopo(raw_path(ETOPO_60S, args.data_dir))}
    print("loading BMNG"); layers["rgb"] = load_bmng(raw_path(bmng_month(args.month), args.data_dir))
    print("loading WorldClim")
    clim = load_worldclim(raw_path(WORLDCLIM_TAVG, args.data_dir), "tavg")
    clim |= load_worldclim(raw_path(WORLDCLIM_PREC, args.data_dir), "prec")
    layers.update({k: clim[k] for k in ("tavg", "trange", "prec", "land")})

    bake(BakeSpec(args.res, layers, nearest={"land"}, flow_downsample=2), out, progress=print)
    print("wrote", out)


if __name__ == "__main__":
    main()
