"""Make per-face model outputs agree across cube-face edges.

Each face is inferred on a raster padded by ``pad`` pixels beyond its edges. For every
interior pixel within ``pad`` of an edge, the neighbouring face's padded prediction is
resampled at this pixel's direction and cross-faded in, with weight 0.5 at the edge
rising to 1.0 for this face's own value ``pad`` pixels inside. Both faces then hold the
same 50/50 mix at the edge, so height and colour are continuous.
"""

from __future__ import annotations

import numpy as np

from .cubesphere import dir_to_face_uv, dir_to_uv_on_face, face_pixel_uv, face_uv_to_dir


def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``a`` (H, W[, C]) at continuous pixel coords (x = col, y = row), clamped."""
    h, w = a.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None] if a.ndim == 3 else x - x0
    fy = (y - y0)[..., None] if a.ndim == 3 else y - y0
    a = a.astype(np.float32)
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def neighbour_face(face: int, edge: str) -> int:
    """Face across ``edge`` ('top' = +v, 'bottom' = -v, 'left' = -u, 'right' = +u)."""
    u, v = {"top": (0.0, 1.001), "bottom": (0.0, -1.001), "left": (-1.001, 0.0), "right": (1.001, 0.0)}[edge]
    return int(dir_to_face_uv(face_uv_to_dir(face, u, v))[0])


def blend_faces(padded: list[np.ndarray], n: int, pad: int) -> list[np.ndarray]:
    """``padded[f]`` is face f's prediction on the (n + 2 pad)^2 padded raster. Returns the
    six (n, n[, C]) face rasters with edge strips cross-faded against the neighbours."""
    out = [p[pad : pad + n, pad : pad + n].astype(np.float32).copy() for p in padded]
    u, v = face_pixel_uv(n)
    rows = np.arange(n)
    for f in range(6):
        for edge in ("top", "bottom", "left", "right"):
            g = neighbour_face(f, edge)
            if edge == "top":
                sl = (slice(0, pad), slice(0, n))
                dist = rows[:pad][:, None].repeat(n, 1)
            elif edge == "bottom":
                sl = (slice(n - pad, n), slice(0, n))
                dist = (n - 1 - rows[n - pad :])[:, None].repeat(n, 1)
            elif edge == "left":
                sl = (slice(0, n), slice(0, pad))
                dist = rows[:pad][None, :].repeat(n, 0)
            else:
                sl = (slice(0, n), slice(n - pad, n))
                dist = (n - 1 - rows[n - pad :])[None, :].repeat(n, 0)
            dirs = face_uv_to_dir(np.full(u[sl].shape, f, dtype=np.intp), u[sl], v[sl])
            gu, gv = dir_to_uv_on_face(dirs, g)
            # padded-raster pixel coordinates on face g
            x = (gu + 1.0) * 0.5 * n - 0.5 + pad
            y = (1.0 - gv) * 0.5 * n - 0.5 + pad
            other = _bilinear(padded[g], x, y)
            w = 0.5 + 0.5 * (dist + 0.5) / pad  # 0.5 at the edge -> 1.0 at `pad` inside
            w = np.clip(w, 0.5, 1.0)
            if out[f].ndim == 3:
                w = w[..., None]
            out[f][sl] = w * out[f][sl] + (1 - w) * other
    return out
