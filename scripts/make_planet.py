#!/usr/bin/env python
"""Generate a procedural planet, colourise it, and write it as a baked cube.

Usage: make_planet.py --seed 3 --res 2048 --checkpoint runs/colorize_adv/last.pt --out data/planets/seed3
The output directory has the same layout as data/cube/<res>, so PlanetMesh can load it.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from planetcreator.colorize import Colorizer, pick_device, tile_infer
from planetcreator.cubesphere import FACE_NAMES
from planetcreator.features import COND_LAYERS, px_km_for_face
from planetcreator.procgen import Planet, PlanetParams, generate_face


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/colorize_adv/last.pt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--land", type=float, default=0.3)
    ap.add_argument("--humidity", type=float, default=1.0)
    ap.add_argument("--equator-temp", type=float, default=28.0)
    ap.add_argument("--pole-temp", type=float, default=-18.0)
    ap.add_argument("--mountains", type=float, default=6000.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-color", action="store_true", help="skip the colouriser (height/climate only)")
    args = ap.parse_args()
    out = args.out or Path("data/planets") / f"seed{args.seed}"
    params = PlanetParams(
        seed=args.seed, land_fraction=args.land, humidity=args.humidity,
        equator_temp_c=args.equator_temp, pole_temp_c=args.pole_temp, mountain_height_m=args.mountains,
    )
    planet = Planet(params)
    model = None if args.no_color else Colorizer.load(args.checkpoint)[0]
    device = pick_device(args.device)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    layer_names = ["height", "lat", "lon", "tavg", "trange", "prec"] + ([] if model is None else ["rgb"])
    t0 = time.time()
    for face, name in enumerate(FACE_NAMES):
        layers = generate_face(planet, face, args.res)
        fdir = out / name
        fdir.mkdir(exist_ok=True)
        for k, v in layers.items():
            np.save(fdir / f"{k}.npy", v)
        np.save(fdir / "holdout.npy", np.zeros((args.res, args.res), dtype=bool))
        if model is not None:
            rgb = tile_infer(model, {k: layers[k] for k in COND_LAYERS}, device=device, px_km=px_km_for_face(args.res))
            np.save(fdir / "rgb.npy", rgb)
            Image.fromarray(rgb[:: max(1, args.res // 1024), :: max(1, args.res // 1024)]).save(out / "preview" / f"{name}_rgb.png")
        h = np.clip((layers["height"] + 6000) / 12000 * 255, 0, 255).astype(np.uint8)
        Image.fromarray(h[:: max(1, args.res // 1024), :: max(1, args.res // 1024)]).save(out / "preview" / f"{name}_height.png")
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)

    meta = {
        "resolution": args.res,
        "faces": list(FACE_NAMES),
        "layers": {k: {"dtype": str(np.load(out / "px" / f"{k}.npy", mmap_mode="r").dtype),
                       "shape": list(np.load(out / "px" / f"{k}.npy", mmap_mode="r").shape)} for k in layer_names},
        "holdout": [],
        "planet": asdict(params),
        "checkpoint": None if model is None else str(args.checkpoint),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
