"""Cube-sphere parameterisation shared by the Python pipeline and the engine.

Conventions (the Rust/Godot side must match these exactly):

* Right-handed axes. +Z is the north pole, +X passes through (lat 0, lon 0),
  +Y through (lat 0, lon 90 E).
* Six faces indexed 0..5 = +X, -X, +Y, -Y, +Z, -Z. Each face has a normal
  ``n`` and tangents ``tu``, ``tv`` with ``tu x tv = n``.
* Face coordinates ``(u, v)`` in [-1, 1]^2. They are tangent-warped
  (``s = tan(u * pi/4)``) before projection, which makes pixel area nearly
  uniform across the face (the S2 "tan" projection).
* A face raster of ``N x N`` pixels has pixel ``(row i, col j)`` centred at
  ``u = (j + 0.5) / N * 2 - 1`` and ``v = 1 - (i + 0.5) / N * 2``:
  row 0 is the +v edge, column 0 the -u edge.
"""

from __future__ import annotations

import numpy as np

# (n, tu, tv) per face; tu x tv == n for every face.
FACE_AXES = np.array(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # +X
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # -X
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # +Y
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # -Y
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # +Z
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # -Z
    ],
    dtype=np.float64,
)
FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")


def warp(u: np.ndarray) -> np.ndarray:
    return np.tan(u * (np.pi / 4))


def unwarp(s: np.ndarray) -> np.ndarray:
    return np.arctan(s) * (4 / np.pi)


def face_uv_to_dir(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unit direction vectors, shape (..., 3)."""
    face = np.asarray(face)
    ax = FACE_AXES[face]  # (..., 3, 3)
    s, t = warp(np.asarray(u, dtype=np.float64)), warp(np.asarray(v, dtype=np.float64))
    d = ax[..., 0, :] + s[..., None] * ax[..., 1, :] + t[..., None] * ax[..., 2, :]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def dir_to_face_uv(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`face_uv_to_dir`. ``d`` need not be normalised."""
    d = np.asarray(d, dtype=np.float64)
    axis = np.argmax(np.abs(d), axis=-1)
    sign = np.take_along_axis(d, axis[..., None], axis=-1)[..., 0] < 0
    face = axis * 2 + sign.astype(np.intp)
    ax = FACE_AXES[face]
    dn = np.einsum("...i,...i->...", d, ax[..., 0, :])
    s = np.einsum("...i,...i->...", d, ax[..., 1, :]) / dn
    t = np.einsum("...i,...i->...", d, ax[..., 2, :]) / dn
    return face, unwarp(s), unwarp(t)


def dir_to_uv_on_face(d: np.ndarray, face: int) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of directions projected onto a *given* face's tangent plane, unclamped
    (|u| > 1 beyond the face edge). Used to resample a neighbouring face's padded raster."""
    d = np.asarray(d, dtype=np.float64)
    n, tu, tv = FACE_AXES[face]
    dn = d @ n
    return unwarp((d @ tu) / dn), unwarp((d @ tv) / dn)


def dir_to_latlon(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degrees. lat in [-90, 90], lon in [-180, 180)."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(d[..., 2] / n, -1, 1)))
    lon = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    lon = np.where(lon >= 180, lon - 360, lon)
    return lat, lon


def latlon_to_dir(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.broadcast_arrays(np.radians(lat), np.radians(lon))
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], axis=-1)


def face_pixel_uv(n: int, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) at every pixel centre of an n x n face raster, each shape (n, n).

    With ``pad`` the raster is (n + 2 pad)^2 and extends beyond the face (|u| > 1); the
    tangent warp continues smoothly there, so the extra pixels land on neighbouring faces.
    """
    c = (np.arange(-pad, n + pad, dtype=np.float64) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(c, -c)  # rows run from +v (top) to -v
    return u, v


def uv_to_pixel(n: int, u: float, v: float) -> tuple[int, int]:
    """Nearest pixel (row, col) of an n x n face raster for face coordinates; clamps at the edges."""
    j = int(np.clip(np.rint((u + 1.0) / 2.0 * n - 0.5), 0, n - 1))
    i = int(np.clip(np.rint((1.0 - v) / 2.0 * n - 0.5), 0, n - 1))
    return i, j


def face_pixel_latlon(face: int, n: int, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    u, v = face_pixel_uv(n, pad)
    return dir_to_latlon(face_uv_to_dir(np.full_like(u, face, dtype=np.intp), u, v))
