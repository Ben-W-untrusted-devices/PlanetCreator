#!/usr/bin/env python
"""Generate a planet: noise base map -> stream-power erosion -> climate -> bake context ->
joint model predicts fine height + RGB per face.

Usage: make_planet.py --seed 3 --checkpoint runs/joint/last.pt --out data/planets/seed3
The output directory has the baked-cube layout, so PlanetMesh can load it.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from planetcreator.bake import CTX_LAYERS, BakeSpec, bake, box_downsample, load_meta
from planetcreator.cubesphere import FACE_NAMES
from planetcreator.erosion import ErosionParams, erode
from planetcreator.features import px_km_for_face
from planetcreator.layers import EquirectGrid
from planetcreator.procgen import (
    Planet,
    PlanetParams,
    climate_equirect,
    generate_equirect,
    uplift_field,
)
from planetcreator.terrain import TerrainNet, coarse_layers_for_face, infer_face, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=4096, help="face resolution of the output cube")
    ap.add_argument("--eq-width", type=int, default=4096, help="equirect width for the base map / erosion")
    ap.add_argument("--iters", type=int, default=200, help="erosion steps")
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/joint/last.pt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--land", type=float, default=0.3)
    ap.add_argument("--humidity", type=float, default=1.0)
    ap.add_argument("--equator-temp", type=float, default=28.0)
    ap.add_argument("--pole-temp", type=float, default=-18.0)
    ap.add_argument("--mountains", type=float, default=6000.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-model", action="store_true", help="stop after baking the coarse planet")
    args = ap.parse_args()
    out = args.out or Path("data/planets") / f"seed{args.seed}"
    t0 = time.time()

    params = PlanetParams(
        seed=args.seed, land_fraction=args.land, humidity=args.humidity,
        equator_temp_c=args.equator_temp, pole_temp_c=args.pole_temp, mountain_height_m=args.mountains,
    )
    planet = Planet(params)
    H, W = args.eq_width // 2, args.eq_width
    print("base map", flush=True)
    g = generate_equirect(planet, H, W)
    print(f"erosion ({args.iters} steps)", flush=True)
    height, _acc = erode(g["height"].astype(np.float64), uplift_field(planet, g["dirs"], g["height"], 1.2),
                         ErosionParams(iters=args.iters), progress=lambda i, n: print(f"  {i}/{n}", flush=True) if i % 50 == 0 else None)
    print("climate", flush=True)
    clim = climate_equirect(planet, g["dirs"], height)

    # The eroded map is the *coarse* truth (context scale); the model invents the fine scale.
    coarse = box_downsample(height, max(1, W // 2048))
    layers = {k: EquirectGrid(v) for k, v in clim.items()}
    print("bake", flush=True)
    bake(BakeSpec(args.res, layers, coarse_height=EquirectGrid(coarse), holdout=()), out)
    np.save(out / "coarse_height_equirect.npy", coarse)
    (out / "preview").mkdir(exist_ok=True)
    step = max(1, args.res // 1024)

    model = None if args.no_model else TerrainNet.load(args.checkpoint)[0]
    device = pick_device(args.device)
    meta = load_meta(out)
    for name in FACE_NAMES:
        fdir = out / name
        ctx = {n: np.load(fdir / f"ctx_{n}.npy") for n in CTX_LAYERS}
        if model is None:
            up = coarse_layers_for_face(ctx, args.res, meta["ctx"]["pad"], meta["ctx"]["factor"])
            np.save(fdir / "height.npy", up["height"].astype(np.float32))
        else:
            fine = {n: np.load(fdir / f"{n}.npy", mmap_mode="r") for n in ("lat", "tavg", "trange", "prec")}
            res = infer_face(model, fine, ctx, device=device, px_km=px_km_for_face(args.res))
            np.save(fdir / "height.npy", res["height"])
            np.save(fdir / "rgb.npy", res["rgb"])
            Image.fromarray(res["rgb"][::step, ::step]).save(out / "preview" / f"{name}_rgb.png")
        h = np.load(fdir / "height.npy", mmap_mode="r")[::step, ::step]
        Image.fromarray(np.clip((np.asarray(h) + 500) / 6000 * 255, 0, 255).astype(np.uint8)).save(out / "preview" / f"{name}_height.png")
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)

    def info(p):
        a = np.load(p, mmap_mode="r")
        return {"dtype": str(a.dtype), "shape": list(a.shape)}

    first = out / FACE_NAMES[0]
    meta["layers"] = {p.stem: info(p) for p in sorted(first.glob("*.npy")) if not p.stem.startswith("ctx_")}
    meta["planet"] = asdict(params)
    meta["erosion"] = {"iters": args.iters, "eq_width": W}
    meta["checkpoint"] = None if model is None else str(args.checkpoint)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
