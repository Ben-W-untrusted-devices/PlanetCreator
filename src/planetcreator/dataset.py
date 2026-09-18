"""Random aligned patches (plus coarse context) from a baked cube for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bake import CTX_LAYERS, DIST_NAMES, load_meta
from .cubesphere import FACE_NAMES

FINE_LAYERS = ("height", "rgb", "lat", "tavg", "trange", "prec", "water", "flowacc", *DIST_NAMES)

# Direction a channel points to, rotated 90 degrees counter-clockwise (np.rot90 convention).
_ROT_CCW = {"dist_up": "dist_left", "dist_left": "dist_down", "dist_down": "dist_right", "dist_right": "dist_up"}
_FLIP_H = {"dist_left": "dist_right", "dist_right": "dist_left"}


def augment_layers(layers: dict[str, np.ndarray], k: int, flip: bool) -> dict[str, np.ndarray]:
    """Rotate every array by 90k degrees CCW and optionally mirror left-right, renaming the
    directional-distance channels so they still point the way they claim to."""
    out = {}
    for name, a in layers.items():
        a = np.rot90(a, k)
        if flip:
            a = a[:, ::-1]
        base, prefix = (name[4:], "ctx_") if name.startswith("ctx_") else (name, "")
        if base in DIST_NAMES:
            for _ in range(k % 4):
                base = _ROT_CCW[base]
            if flip:
                base = _FLIP_H.get(base, base)
        out[prefix + base] = np.ascontiguousarray(a)
    return out


@dataclass
class PatchSpec:
    patch: int = 256
    layers: tuple[str, ...] = FINE_LAYERS
    ctx: bool = True  # also return the coarse context window (ctx_<layer>)
    split: str = "train"  # "train" (no holdout pixels) or "val" (only holdout pixels)
    min_land_frac: float = 0.0  # reject patches with less land than this (uses height > 0)
    augment: bool = True  # random 90-degree rotations and flips
    max_tries: int = 200


class CubePatchDataset(Dataset):
    """Yields ``{layer: tensor(C, P, P), ctx_<layer>: tensor(C, Pc, Pc), meta}``.

    Deterministic per ``(seed, index)``. Patch origins are multiples of the context factor
    so the context window ``[i - pad, i + P + pad)`` maps exactly onto context pixels.
    Patches never cross a face edge; the context does (it is baked padded).
    """

    def __init__(self, root: Path, spec: PatchSpec, length: int = 100_000, seed: int = 0):
        self.root, self.spec, self.length, self.seed = Path(root), spec, length, seed
        self.meta = load_meta(self.root)
        self.n = self.meta["resolution"]
        self.f = self.meta["ctx"]["factor"]
        self.pad = self.meta["ctx"]["pad"]
        if spec.patch > self.n or spec.patch % self.f:
            raise ValueError(f"patch {spec.patch} must be <= face {self.n} and a multiple of {self.f}")
        self.ctx_size = (spec.patch + 2 * self.pad) // self.f
        self._maps: dict[tuple[str, str], np.memmap] = {}
        self._val_boxes: list[tuple[str, int, int, int, int]] = []
        if spec.split == "val":
            for face in FACE_NAMES:
                rows, cols = np.nonzero(np.load(self.root / face / "holdout.npy", mmap_mode="r"))
                if len(rows):
                    self._val_boxes.append((face, rows.min(), rows.max(), cols.min(), cols.max()))
            if not self._val_boxes:
                raise ValueError("no holdout pixels in this cube; val split is empty")

    def _layer(self, face: str, name: str) -> np.ndarray:
        key = (face, name)
        if key not in self._maps:
            self._maps[key] = np.load(self.root / face / f"{name}.npy", mmap_mode="r")
        return self._maps[key]

    def __len__(self) -> int:
        return self.length

    def _accept(self, face: str, i: int, j: int) -> bool:
        p = self.spec.patch
        hold = self._layer(face, "holdout")[i : i + p, j : j + p]
        if self.spec.split == "train" and hold.any():
            return False
        if self.spec.split == "val" and not hold.all():
            return False
        if self.spec.min_land_frac > 0:
            h = self._layer(face, "height")[i : i + p, j : j + p]
            if (h > 0).mean() < self.spec.min_land_frac:
                return False
        return True

    def _origin(self, rng: np.random.Generator) -> tuple[str, int, int]:
        p, f = self.spec.patch, self.f
        if self.spec.split == "val":
            face, r0, r1, c0, c1 = self._val_boxes[rng.integers(len(self._val_boxes))]
            i = rng.integers(max(r0 - p + 1, 0) // f, min(r1, self.n - p) // f + 1) * f
            j = rng.integers(max(c0 - p + 1, 0) // f, min(c1, self.n - p) // f + 1) * f
        else:
            face = FACE_NAMES[rng.integers(6)]
            i, j = rng.integers(0, (self.n - p) // f + 1, size=2) * f
        return face, int(i), int(j)

    def raw(self, face: str, i: int, j: int) -> dict[str, np.ndarray]:
        """Un-augmented layers for a patch at (face, i, j)."""
        p = self.spec.patch
        out = {name: np.asarray(self._layer(face, name)[i : i + p, j : j + p]) for name in self.spec.layers}
        if self.spec.ctx:
            ci, cj, s = i // self.f, j // self.f, self.ctx_size
            for name in CTX_LAYERS:
                out[f"ctx_{name}"] = np.asarray(self._layer(face, f"ctx_{name}")[ci : ci + s, cj : cj + s])
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng([self.seed, idx])
        for _ in range(self.spec.max_tries):
            face, i, j = self._origin(rng)
            if self._accept(face, i, j):
                break
        else:
            raise RuntimeError("could not find an acceptable patch; relax PatchSpec constraints")

        k = int(rng.integers(4)) if self.spec.augment else 0
        flip = bool(rng.integers(2)) if self.spec.augment else False
        out = {}
        for name, a in augment_layers(self.raw(face, i, j), k, flip).items():
            if a.ndim == 2:
                a = a[..., None]
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
            out[name] = t.float() / 255 if a.dtype == np.uint8 else t.float()
        out["meta"] = torch.tensor([FACE_NAMES.index(face), i, j, k, int(flip)])
        return out
