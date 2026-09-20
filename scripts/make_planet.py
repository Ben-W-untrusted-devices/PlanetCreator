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
from planetcreator.faceblend import blend_faces
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

FINE_PAD = 128  # fine pixels inferred beyond each face edge, then cross-faded with the neighbour


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=4096, help="face resolution of the output cube")
    ap.add_argument("--eq-width", type=int, default=4096, help="equirect width for the base map / erosion")
    ap.add_argument("--iters", type=int, default=200, help="erosion steps")
    ap.add_argument("--smooth", type=float, nargs=2, default=(6.0, 3.0), metavar=("LOWLAND", "MOUNTAIN"),
                    help="Gaussian sigmas (erosion px) blended by elevation before downsampling")
    ap.add_argument("--checkpoint", type=Path, default=Path("models/joint_latest.pt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--land", type=float, default=0.3)
    ap.add_argument("--humidity", type=float, default=1.0)
    ap.add_argument("--equator-temp", type=float, default=28.0)
    ap.add_argument("--pole-temp", type=float, default=-18.0)
    ap.add_argument("--mountains", type=float, default=6000.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-model", action="store_true", help="stop after baking the coarse planet")
    ap.add_argument("--reuse", action="store_true", help="skip generation; run the model on an existing bake")
    args = ap.parse_args()
    out = args.out or Path("data/planets") / f"seed{args.seed}"
    t0 = time.time()

    params = PlanetParams(
        seed=args.seed, land_fraction=args.land, humidity=args.humidity,
        equator_temp_c=args.equator_temp, pole_temp_c=args.pole_temp, mountain_height_m=args.mountains,
    )
    planet = Planet(params)
    H, W = args.eq_width // 2, args.eq_width
    if args.reuse and (out / "meta.json").exists():
        print("reusing bake in", out, flush=True)
        infer_faces(args, out, t0)
        return
    print("base map", flush=True)
    g = generate_equirect(planet, H, W)
    print(f"erosion ({args.iters} steps)", flush=True)
    height, _acc = erode(g["height"].astype(np.float64), uplift_field(planet, g["dirs"], g["height"], 1.2),
                         ErosionParams(iters=args.iters, routing_power=1.0, k_variation=0.4), progress=lambda i, n: print(f"  {i}/{n}", flush=True) if i % 50 == 0 else None)
    # The eroded map is the *coarse* truth (context scale); the model invents the fine scale.
    # Its roughness must match Earth's coarse map (smooth lowlands, rugged belts; see
    # docs/approaches.md) or the model amplifies grid-scale dissection into hatching.
    # Elevation-blended Gaussian smoothing gets the gradient statistics into Earth's range.
    if max(args.smooth) > 0:
        from scipy.ndimage import gaussian_filter

        ocean = height <= 0
        w = np.clip(height / 1500.0, 0, 1)
        low = gaussian_filter(height, args.smooth[0], mode=("nearest", "wrap"))
        mtn = gaussian_filter(height, args.smooth[1], mode=("nearest", "wrap"))
        height = np.where(ocean, height, np.maximum((1 - w) * low + w * mtn, 0.5)).astype(np.float32)
    # Climate is derived from the *smoothed* height: temperature carries altitude (lapse
    # rate), so anything finer than the coarse map here would leak unsmoothed erosion
    # structure into the model's fine inputs.
    print("climate", flush=True)
    clim = climate_equirect(planet, g["dirs"], height)
    coarse = box_downsample(height, max(1, W // 2048))
    layers = {k: EquirectGrid(v) for k, v in clim.items()}
    print("bake", flush=True)
    # Fine layers padded by FINE_PAD so faces can be inferred past their edges and blended;
    # the context must then be padded by the same extra amount.
    bake(BakeSpec(args.res, layers, coarse_height=EquirectGrid(coarse), holdout=(), fine_pad=FINE_PAD, ctx_pad=896 + FINE_PAD), out)
    np.save(out / "coarse_height_equirect.npy", coarse)
    meta = load_meta(out)
    meta["planet"] = asdict(params)
    meta["erosion"] = {"iters": args.iters, "eq_width": W}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    infer_faces(args, out, t0)


def infer_faces(args, out: Path, t0: float) -> None:
    (out / "preview").mkdir(exist_ok=True)
    step = max(1, args.res // 1024)
    model = None if args.no_model else TerrainNet.load(args.checkpoint)[0]
    device = pick_device(args.device)
    meta = load_meta(out)
    fpad = meta.get("fine_pad", 0)
    ctx_stored = meta["ctx"]["pad"]
    preds: dict[str, list[np.ndarray]] = {"height": [], "rgb": []}
    for face, name in enumerate(FACE_NAMES):
        fdir = out / name
        ctx = {n: np.load(fdir / f"ctx_{n}.npy") for n in CTX_LAYERS}
        if model is None:
            up = coarse_layers_for_face(ctx, args.res, meta["ctx"]["pad"], meta["ctx"]["factor"])
            np.save(fdir / "height.npy", up["height"].astype(np.float32))
        else:
            prefix = "pad_" if fpad else ""
            fine = {n: np.load(fdir / f"{prefix}{n}.npy", mmap_mode="r") for n in ("lat", "tavg", "trange", "prec")}
            res = infer_face(model, fine, ctx, device=device, px_km=px_km_for_face(args.res), face=face, seed=args.seed,
                             origin_i=-fpad, origin_j=-fpad, ctx_stored_pad=ctx_stored)
            preds["height"].append(res["height"])
            preds["rgb"].append(res["rgb"])
        print(f"{name} inferred ({time.time() - t0:.0f}s)", flush=True)
    if model is not None:
        if fpad:
            print("blending face edges", flush=True)
            heights = blend_faces(preds["height"], args.res, fpad)
            rgbs = blend_faces(preds["rgb"], args.res, fpad)
        else:
            heights, rgbs = preds["height"], preds["rgb"]
        for face, name in enumerate(FACE_NAMES):
            fdir = out / name
            np.save(fdir / "height.npy", heights[face].astype(np.float32))
            np.save(fdir / "rgb.npy", np.clip(np.rint(rgbs[face]), 0, 255).astype(np.uint8))
            Image.fromarray(np.load(fdir / "rgb.npy")[::step, ::step]).save(out / "preview" / f"{name}_rgb.png")
    for name in FACE_NAMES:
        fdir = out / name
        h = np.load(fdir / "height.npy", mmap_mode="r")[::step, ::step]
        Image.fromarray(np.clip((np.asarray(h) + 500) / 6000 * 255, 0, 255).astype(np.uint8)).save(out / "preview" / f"{name}_height.png")
        # 1:1 hillshade of the centre of the face: block/grid artefacts are invisible in
        # a downsampled height preview but obvious here
        c = args.res // 2
        full = np.asarray(np.load(fdir / "height.npy", mmap_mode="r")[c - 400 : c + 400, c - 400 : c + 400], dtype=np.float64)
        gy, gx = np.gradient(full)
        shade = np.clip(0.5 + (gx - gy) / 250.0, 0, 1)
        img = np.stack([shade] * 3, -1)
        img[full <= 0] = [0.05, 0.1, 0.25]
        Image.fromarray((img * 255).astype(np.uint8)).save(out / "preview" / f"{name}_hillshade.png")
    print(f"done ({time.time() - t0:.0f}s)", flush=True)

    def info(p):
        a = np.load(p, mmap_mode="r")
        return {"dtype": str(a.dtype), "shape": list(a.shape)}

    first = out / FACE_NAMES[0]
    meta["layers"] = {p.stem: info(p) for p in sorted(first.glob("*.npy")) if not p.stem.startswith("ctx_")}
    meta["checkpoint"] = None if model is None else str(args.checkpoint)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
