"""Vectorised 3D Perlin noise and fBm variants, evaluated at points on the unit sphere.

Sampling noise in 3D at each face pixel's direction gives seamless cube-sphere
fields for free: adjacent faces evaluate the same function at the same points.
"""

from __future__ import annotations

import numpy as np

_GRADS = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
     [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
    dtype=np.float32,
)


class Perlin3:
    def __init__(self, seed: int):
        p = np.random.default_rng(seed).permutation(256).astype(np.int32)
        self.perm = np.concatenate([p, p])

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def __call__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Noise in roughly [-1, 1] at float32 coordinates of any shape."""
        xi, yi, zi = (np.floor(v).astype(np.int32) for v in (x, y, z))
        xf, yf, zf = x - xi, y - yi, z - zi
        xi, yi, zi = xi & 255, yi & 255, zi & 255
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        p = self.perm

        def grad(hx, hy, hz, dx, dy, dz):
            h = p[p[p[hx] + hy] + hz] % 12
            g = _GRADS[h]
            return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

        def lerp(a, b, t):
            return a + t * (b - a)

        x1 = lerp(grad(xi, yi, zi, xf, yf, zf), grad(xi + 1, yi, zi, xf - 1, yf, zf), u)
        x2 = lerp(grad(xi, yi + 1, zi, xf, yf - 1, zf), grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf), u)
        y1 = lerp(x1, x2, v)
        x1 = lerp(grad(xi, yi, zi + 1, xf, yf, zf - 1), grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1), u)
        x2 = lerp(grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1), grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1), u)
        y2 = lerp(x1, x2, v)
        return lerp(y1, y2, w)


def fbm(noise: Perlin3, p: np.ndarray, freq: float, octaves: int, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Fractal sum; ``p`` is (..., 3). Output roughly in [-1, 1]."""
    out = np.zeros(p.shape[:-1], dtype=np.float32)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        q = p * freq
        out += amp * noise(q[..., 0], q[..., 1], q[..., 2])
        total += amp
        amp *= gain
        freq *= lacunarity
    return out / total


def ridged(noise: Perlin3, p: np.ndarray, freq: float, octaves: int, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Ridged multifractal in [0, 1]: sharp crests, good for mountain ranges."""
    out = np.zeros(p.shape[:-1], dtype=np.float32)
    amp, total, weight = 1.0, 0.0, np.ones(p.shape[:-1], dtype=np.float32)
    for _ in range(octaves):
        q = p * freq
        r = (1.0 - np.abs(noise(q[..., 0], q[..., 1], q[..., 2]))) ** 2 * weight
        weight = np.clip(r * 2.0, 0, 1)
        out += amp * r
        total += amp
        amp *= gain
        freq *= lacunarity
    return out / total


def warp(noise: Perlin3, p: np.ndarray, freq: float, strength: float, seed_offset: float = 17.3) -> np.ndarray:
    """Domain warp: displace points by a low-frequency vector field. Breaks up noise's blobbiness."""
    q = p * freq
    d = np.stack(
        [noise(q[..., 0] + k * seed_offset, q[..., 1] + k * seed_offset, q[..., 2] + k * seed_offset) for k in range(3)],
        axis=-1,
    )
    return p + strength * d
