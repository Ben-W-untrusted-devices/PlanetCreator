"""Hydrology on a global equirectangular grid: depression filling, D8 routing, flow
accumulation, lake detection. Longitude wraps; rows are clamped at the poles.

Used for two things: derived conditioning channels (Earth and synthetic alike) and
the inner loop of the erosion simulator. All heavy loops are numba-compiled.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

EARTH_RADIUS_KM = 6371.0
_DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_DIST = np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)], dtype=np.float64)


# ---------------------------------------------------------------------------------------
# binary min-heap on flat arrays (numba-friendly)


@njit(cache=True)
def _heap_push(keys, vals, size, key, val):
    i = size
    keys[i] = key
    vals[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if keys[p] <= keys[i]:
            break
        keys[p], keys[i] = keys[i], keys[p]
        vals[p], vals[i] = vals[i], vals[p]
        i = p
    return size + 1


@njit(cache=True)
def _heap_pop(keys, vals, size):
    key, val = keys[0], vals[0]
    size -= 1
    keys[0], vals[0] = keys[size], vals[size]
    i = 0
    while True:
        l, r = 2 * i + 1, 2 * i + 2
        m = i
        if l < size and keys[l] < keys[m]:
            m = l
        if r < size and keys[r] < keys[m]:
            m = r
        if m == i:
            break
        keys[m], keys[i] = keys[i], keys[m]
        vals[m], vals[i] = vals[i], vals[m]
        i = m
    return key, val, size


# ---------------------------------------------------------------------------------------


@njit(cache=True)
def priority_flood(h, sea_level, eps):
    """Barnes et al. 2014 priority-flood: fill depressions so every land cell drains to the
    sea, with a tiny gradient ``eps`` across flats. Cells at or below ``sea_level`` are outlets.
    Returns the filled surface."""
    H, W = h.shape
    n = H * W
    filled = h.copy()
    closed = np.zeros(n, dtype=np.uint8)
    keys = np.empty(n, dtype=np.float64)
    vals = np.empty(n, dtype=np.int64)
    size = 0
    for i in range(H):
        for j in range(W):
            c = i * W + j
            if h[i, j] <= sea_level:
                closed[c] = 1
                # seed the queue from ocean cells that touch land
                for k in range(8):
                    ni = i + _DI[k]
                    nj = (j + _DJ[k]) % W
                    if ni < 0 or ni >= H:
                        continue
                    if h[ni, nj] > sea_level:
                        size = _heap_push(keys, vals, size, h[i, j], c)
                        break
            elif i == 0 or i == H - 1:
                size = _heap_push(keys, vals, size, h[i, j], c)
                closed[c] = 1
    while size > 0:
        key, c, size = _heap_pop(keys, vals, size)
        i, j = c // W, c % W
        for k in range(8):
            ni = i + _DI[k]
            nj = (j + _DJ[k]) % W
            if ni < 0 or ni >= H:
                continue
            nc = ni * W + nj
            if closed[nc]:
                continue
            closed[nc] = 1
            filled[ni, nj] = max(filled[ni, nj], key + eps)
            size = _heap_push(keys, vals, size, filled[ni, nj], nc)
    return filled


@njit(cache=True)
def row_cos(H, min_cos):
    """cos(latitude) per row of an equirect grid, floored so polar cells stay well-posed."""
    out = np.empty(H)
    for i in range(H):
        lat = (90.0 - (i + 0.5) * 180.0 / H) * np.pi / 180.0
        out[i] = max(np.cos(lat), min_cos)
    return out


@njit(cache=True)
def _step_len(k, cos_i):
    """Distance to neighbour k in units of the row spacing, for a row with E-W scale cos_i."""
    di = abs(_DI[k])
    dj = abs(_DJ[k]) * cos_i
    return np.sqrt(di * di + dj * dj)


@njit(cache=True)
def d8_receivers(filled, sea_level, cosr):
    """Steepest-descent receiver (flat index) per cell; sinks and ocean point to themselves.
    ``cosr`` is the per-row E-W spacing factor (see :func:`row_cos`)."""
    H, W = filled.shape
    rec = np.empty(H * W, dtype=np.int64)
    for i in range(H):
        for j in range(W):
            c = i * W + j
            rec[c] = c
            if filled[i, j] <= sea_level:
                continue
            best = 0.0
            for k in range(8):
                ni = i + _DI[k]
                nj = (j + _DJ[k]) % W
                if ni < 0 or ni >= H:
                    continue
                s = (filled[i, j] - filled[ni, nj]) / _step_len(k, cosr[i])
                if s > best:
                    best = s
                    rec[c] = ni * W + nj
    return rec


@njit(cache=True)
def _hash01(a, b):
    """Cheap deterministic uniform in [0, 1) from two integers."""
    x = (a * 73856093) ^ (b * 19349663)
    x = (x ^ (x >> 13)) * 1274126177
    x = x ^ (x >> 16)
    return (x & 0xFFFFFF) / 16777216.0


@njit(cache=True)
def d8_receivers_stochastic(filled, sea_level, power, salt, cosr):
    """Like :func:`d8_receivers` but the receiver is drawn among all downslope neighbours
    with probability proportional to slope**power. Breaks the straight parallel channels
    deterministic D8 carves on smooth slopes; ``salt`` changes the draw per call."""
    H, W = filled.shape
    rec = np.empty(H * W, dtype=np.int64)
    slopes = np.empty(8)
    targets = np.empty(8, dtype=np.int64)
    for i in range(H):
        for j in range(W):
            c = i * W + j
            rec[c] = c
            if filled[i, j] <= sea_level:
                continue
            total = 0.0
            n = 0
            for k in range(8):
                ni = i + _DI[k]
                nj = (j + _DJ[k]) % W
                if ni < 0 or ni >= H:
                    continue
                sl = (filled[i, j] - filled[ni, nj]) / _step_len(k, cosr[i])
                if sl > 0:
                    w = sl**power
                    slopes[n] = w
                    targets[n] = ni * W + nj
                    total += w
                    n += 1
            if n == 0:
                continue
            r = _hash01(c, salt) * total
            acc = 0.0
            for k in range(n):
                acc += slopes[k]
                if r < acc:
                    rec[c] = targets[k]
                    break
            else:
                rec[c] = targets[n - 1]
    return rec


@njit(cache=True)
def flow_order(rec):
    """Topological order, upstream cells first (Kahn's algorithm over the receiver graph)."""
    n = rec.shape[0]
    ndon = np.zeros(n, dtype=np.int32)
    for c in range(n):
        if rec[c] != c:
            ndon[rec[c]] += 1
    order = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for c in range(n):
        if ndon[c] == 0:
            order[tail] = c
            tail += 1
    while head < tail:
        c = order[head]
        head += 1
        r = rec[c]
        if r != c:
            ndon[r] -= 1
            if ndon[r] == 0:
                order[tail] = r
                tail += 1
    return order[:tail]


@njit(cache=True)
def accumulate(rec, order, area):
    """Upstream area per cell (same units as ``area``), summed in flow order."""
    acc = area.copy()
    for k in range(order.shape[0]):
        c = order[k]
        r = rec[c]
        if r != c:
            acc[r] += acc[c]
    return acc


# ---------------------------------------------------------------------------------------


def cell_area_km2(H: int, W: int) -> np.ndarray:
    """Flat (H*W,) cell areas of an equirectangular grid on an Earth-radius sphere."""
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)
    dlat = np.pi / H
    dlon = 2 * np.pi / W
    row = EARTH_RADIUS_KM**2 * dlat * dlon * np.cos(lat)
    return np.repeat(row, W).astype(np.float64)


def route(height: np.ndarray, sea_level: float = 0.0, eps: float = 1e-3, min_cos: float = 0.15):
    """Fill, route and accumulate. Returns (filled, receivers, order, accumulation_km2)."""
    h = np.ascontiguousarray(height, dtype=np.float64)
    filled = priority_flood(h, sea_level, eps)
    rec = d8_receivers(filled, sea_level, row_cos(h.shape[0], min_cos))
    order = flow_order(rec)
    acc = accumulate(rec, order, cell_area_km2(*h.shape))
    return filled, rec, order, acc.reshape(h.shape)


def lake_mask(
    height: np.ndarray,
    filled: np.ndarray,
    sea_level: float = 0.0,
    min_fill_m: float = 10.0,
    flat_tol_m: float = 0.05,
    flat_min_cells: int = 40,
) -> np.ndarray:
    """Inland water: cells inside filled depressions, or in sizeable dead-flat regions.

    Lake surfaces in a DEM are *exactly* flat but not necessarily depressions; the
    tolerance must stay well below floodplain gradients (the Amazon varies by decimetres
    over a few km) or lowlands get tagged as lakes."""
    land = height > sea_level
    lakes = land & (filled - height > min_fill_m)
    rng = ndimage.maximum_filter(height, 5) - ndimage.minimum_filter(height, 5)
    flat = land & (rng < flat_tol_m)
    labels, n = ndimage.label(flat)
    if n:
        sizes = ndimage.sum(flat, labels, index=np.arange(1, n + 1))
        big = np.zeros(n + 1, dtype=bool)
        big[1:] = sizes >= flat_min_cells
        lakes |= big[labels]
    return lakes


@njit(cache=True)
def _dist_along_rows(water, cap, wrap):
    """Distance in cells from each cell to the nearest water cell at a higher column index
    (looking right/east), optionally wrapping around the row."""
    H, W = water.shape
    out = np.full((H, W), cap, dtype=np.float32)
    span = 2 * W if wrap else W
    for i in range(H):
        nxt = -1  # column index (in the doubled row) of the next water cell to the right
        for jj in range(span - 1, -1, -1):
            j = jj % W
            if water[i, j]:
                nxt = jj
                out[i, j] = 0.0
            elif jj < W and nxt >= 0:
                d = nxt - jj
                out[i, j] = min(out[i, j], d)
    return out


def directional_distance(
    water: np.ndarray, spacing_km: np.ndarray | float, cap_km: float, axis: int, sign: int, wrap: bool = True
) -> np.ndarray:
    """Distance (km) from each cell to the nearest water cell looking along ``axis`` in
    direction ``sign``; wraps along axis 1 if ``wrap``, clamps along axis 0; capped at ``cap_km``.

    ``spacing_km`` is a scalar or a per-row array (E/W spacing shrinks with latitude).
    Distances are measured in cells then scaled by the local spacing.
    """
    w = np.ascontiguousarray(water, dtype=np.uint8)
    big = np.float32(10**9)
    if axis == 0:
        src = np.ascontiguousarray(w.T[:, ::-1] if sign < 0 else w.T)
        d = _dist_along_rows(src, big, False)
        d = (d[:, ::-1] if sign < 0 else d).T
    else:
        src = np.ascontiguousarray(w[:, ::-1] if sign < 0 else w)
        d = _dist_along_rows(src, big, wrap)
        d = d[:, ::-1] if sign < 0 else d
    sp = np.asarray(spacing_km, dtype=np.float32)
    if sp.ndim == 1:
        sp = sp[:, None]
    return np.minimum(d * sp, cap_km).astype(np.float32)
