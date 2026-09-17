"""Random aligned patches from a baked cube for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bake import load_meta
from .cubesphere import FACE_NAMES


@dataclass
class PatchSpec:
    patch: int = 256
    layers: tuple[str, ...] = ("height", "rgb", "lat", "tavg", "trange", "prec")
    split: str = "train"  # "train" (no holdout pixels) or "val" (only holdout pixels)
    min_land_frac: float = 0.0  # reject patches with less land than this (uses height > 0)
    augment: bool = True  # random 90-degree rotations and flips
    max_tries: int = 200


class CubePatchDataset(Dataset):
    """Yields ``{layer: tensor(C, P, P)}`` patches. Deterministic per ``(seed, index)``.

    Patches never cross a face edge; the small loss of coverage at edges is
    acceptable since faces overlap neither in data nor in holdout logic.
    """

    def __init__(self, root: Path, spec: PatchSpec, length: int = 100_000, seed: int = 0):
        self.root, self.spec, self.length, self.seed = Path(root), spec, length, seed
        self.meta = load_meta(self.root)
        self.n = self.meta["resolution"]
        if spec.patch > self.n:
            raise ValueError(f"patch {spec.patch} larger than face {self.n}")
        self._maps: dict[tuple[str, str], np.memmap] = {}
        # For the val split, restrict sampling to each face's holdout bounding box so
        # small held-out regions are found without thousands of rejections.
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng([self.seed, idx])
        p = self.spec.patch
        for _ in range(self.spec.max_tries):
            if self.spec.split == "val":
                face, r0, r1, c0, c1 = self._val_boxes[rng.integers(len(self._val_boxes))]
                i = rng.integers(max(r0 - p + 1, 0), min(r1, self.n - p) + 1)
                j = rng.integers(max(c0 - p + 1, 0), min(c1, self.n - p) + 1)
            else:
                face = FACE_NAMES[rng.integers(6)]
                i, j = rng.integers(0, self.n - p + 1, size=2)
            if self._accept(face, i, j):
                break
        else:
            raise RuntimeError("could not find an acceptable patch; relax PatchSpec constraints")

        k = rng.integers(4) if self.spec.augment else 0
        flip = bool(rng.integers(2)) if self.spec.augment else False
        out = {}
        for name in self.spec.layers:
            a = np.asarray(self._layer(face, name)[i : i + p, j : j + p])
            if a.ndim == 2:
                a = a[..., None]
            a = np.rot90(a, k)
            if flip:
                a = a[:, ::-1]
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
            out[name] = t.float() / 255 if a.dtype == np.uint8 else t.float()
        out["meta"] = torch.tensor([FACE_NAMES.index(face), i, j, k, int(flip)])
        return out
