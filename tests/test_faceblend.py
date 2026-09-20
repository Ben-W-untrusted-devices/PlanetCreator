import numpy as np

from planetcreator.cubesphere import (
    dir_to_face_uv,
    face_pixel_uv,
    face_uv_to_dir,
    latlon_to_dir,
    uv_to_pixel,
)
from planetcreator.faceblend import blend_faces, neighbour_face


def test_neighbours():
    assert neighbour_face(0, "right") == 2  # +X's +u edge meets +Y
    assert neighbour_face(0, "left") == 3  # and -Y on the other side
    assert neighbour_face(0, "top") == 4  # +Z above
    assert neighbour_face(2, "left") == 0


def test_blend_makes_smooth_function_continuous_and_leaves_interior():
    n, pad = 64, 8
    f = lambda d: 1000.0 * np.sin(2 * d[..., 0]) * np.cos(3 * d[..., 1]) + 500 * d[..., 2]
    padded = []
    for face in range(6):
        u, v = face_pixel_uv(n, pad)
        d = face_uv_to_dir(np.full(u.shape, face, dtype=np.intp), u, v)
        val = f(d).astype(np.float32)
        # each face "hallucinates" a different offset: the seam we want blended away
        padded.append(val + 100.0 * face)
    out = blend_faces(padded, n, pad)
    # interior untouched
    assert np.allclose(out[0][pad:-pad, pad:-pad], padded[0][2 * pad : -2 * pad, 2 * pad : -2 * pad])
    # walk across the +X/+Y edge: values on both sides agree at the edge (up to bilinear error)
    lat, lon0, lon1 = 5.0, 44.9, 45.1
    dA = latlon_to_dir(lat, lon0)
    dB = latlon_to_dir(lat, lon1)
    fa, ua, va = dir_to_face_uv(dA)
    fb, ub, vb = dir_to_face_uv(dB)
    assert (fa, fb) == (0, 2)
    ia, ja = uv_to_pixel(n, ua, va)
    ib, jb = uv_to_pixel(n, ub, vb)
    jump = abs(out[0][ia, ja] - out[2][ib, jb])
    raw_jump = abs(padded[0][ia + pad, ja + pad] - padded[2][ib + pad, jb + pad])
    assert raw_jump > 150 and jump < 40, (raw_jump, jump)
