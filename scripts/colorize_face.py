#!/usr/bin/env python
"""Colourise a whole cube face: colorize_face.py runs/colorize/last.pt --face ny [--cube data/cube/4096]

Writes <out>/<face>_pred.png next to a <face>_true.png for comparison.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from planetcreator.colorize import Colorizer, pick_device, tile_infer
from planetcreator.features import COND_LAYERS, px_km_for_face


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--cube", type=Path, default=Path("data/cube/4096"))
    ap.add_argument("--face", default="ny")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--downsample", type=int, default=2, help="stride to keep inference quick")
    ap.add_argument("--sea-level", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = args.out or args.checkpoint.parent
    d = args.downsample
    layers = {n: np.load(args.cube / args.face / f"{n}.npy", mmap_mode="r")[::d, ::d] for n in COND_LAYERS}
    model, _ = Colorizer.load(args.checkpoint)
    res = layers["height"].shape[0]
    rgb = tile_infer(model, layers, device=pick_device(args.device), sea_level=args.sea_level, px_km=px_km_for_face(res))
    Image.fromarray(rgb).save(out / f"{args.face}_pred.png")
    true = np.load(args.cube / args.face / "rgb.npy", mmap_mode="r")[::d, ::d]
    Image.fromarray(np.asarray(true)).save(out / f"{args.face}_true.png")
    print("wrote", out / f"{args.face}_pred.png")


if __name__ == "__main__":
    main()
