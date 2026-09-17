#!/usr/bin/env python
"""Write PNG previews of a baked cube: preview_cube.py data/cube/4096 [--size 1024]"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from planetcreator.bake import load_meta
from planetcreator.cubesphere import FACE_NAMES


def to_u8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    meta = load_meta(args.root)
    out = args.out or args.root / "preview"
    out.mkdir(exist_ok=True)
    step = max(1, meta["resolution"] // args.size)
    for face in FACE_NAMES:
        for name in meta["layers"]:
            a = np.load(args.root / face / f"{name}.npy", mmap_mode="r")[::step, ::step]
            if a.dtype == np.uint8:
                img = np.asarray(a)
            elif name == "height":
                img = to_u8(np.asarray(a), -8000, 8000)
            elif a.dtype == bool:
                img = np.asarray(a).astype(np.uint8) * 255
            else:
                a = np.asarray(a); img = to_u8(a, np.percentile(a, 1), np.percentile(a, 99))
            Image.fromarray(img).save(out / f"{face}_{name}.png")
        hold = np.load(args.root / face / "holdout.npy", mmap_mode="r")[::step, ::step]
        Image.fromarray(np.asarray(hold).astype(np.uint8) * 255).save(out / f"{face}_holdout.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
