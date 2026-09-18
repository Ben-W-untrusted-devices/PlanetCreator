#!/usr/bin/env python
"""Run a trained model over a whole cube face: render_face.py runs/joint/last.pt --face ny

Writes <out>/<face>_rgb.png (and _height.png in joint mode) next to the truth for comparison.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from planetcreator.bake import CTX_LAYERS, DIST_NAMES, load_meta
from planetcreator.features import px_km_for_face
from planetcreator.terrain import TerrainNet, infer_face, pick_device


def height_png(h: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip((h + 500) / 6000 * 255, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--cube", type=Path, default=Path("data/cube/4096"))
    ap.add_argument("--face", default="ny")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = args.out or args.checkpoint.parent
    meta = load_meta(args.cube)
    model, _ = TerrainNet.load(args.checkpoint)
    fdir = args.cube / args.face
    names = ("lat", "tavg", "trange", "prec") + (("height", "water", "flowacc", *DIST_NAMES) if model.cfg.mode == "colour" else ())
    fine = {n: np.load(fdir / f"{n}.npy", mmap_mode="r") for n in names}
    ctx = {n: np.load(fdir / f"ctx_{n}.npy", mmap_mode="r") for n in CTX_LAYERS}
    res = infer_face(model, fine, ctx, device=pick_device(args.device), px_km=px_km_for_face(meta["resolution"]))
    Image.fromarray(res["rgb"]).save(out / f"{args.face}_rgb.png")
    if (fdir / "rgb.npy").exists():
        Image.fromarray(np.asarray(np.load(fdir / "rgb.npy", mmap_mode="r"))).save(out / f"{args.face}_rgb_true.png")
    if "height" in res:
        height_png(res["height"]).save(out / f"{args.face}_height.png")
        if (fdir / "height.npy").exists():
            height_png(np.asarray(np.load(fdir / "height.npy", mmap_mode="r"))).save(out / f"{args.face}_height_true.png")
    print("wrote", out / f"{args.face}_rgb.png")


if __name__ == "__main__":
    main()
