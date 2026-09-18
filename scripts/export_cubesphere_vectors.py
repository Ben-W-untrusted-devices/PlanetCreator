#!/usr/bin/env python
"""Export cube-sphere test vectors so the Rust implementation can be checked against Python.

Writes rust/crates/planet-core/tests/data/cubesphere_vectors.json. Re-run and commit
whenever cubesphere.py changes (it should not change often).
"""

import json
from pathlib import Path

import numpy as np

from planetcreator import cubesphere as cs

OUT = Path(__file__).resolve().parents[1] / "rust/crates/planet-core/tests/data/cubesphere_vectors.json"


def main() -> None:
    rng = np.random.default_rng(42)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    face, u, v = cs.dir_to_face_uv(d)
    lat, lon = cs.dir_to_latlon(d)
    records = [
        {"dir": d[i].tolist(), "face": int(face[i]), "u": float(u[i]), "v": float(v[i]),
         "lat": float(lat[i]), "lon": float(lon[i])}
        for i in range(len(d))
    ]
    n = 8
    pu, pv = cs.face_pixel_uv(n)
    pixels = []
    for f in range(6):
        plat, plon = cs.face_pixel_latlon(f, n)
        for i, j in ((0, 0), (0, n - 1), (n - 1, 0), (3, 5)):
            pixels.append({"face": f, "n": n, "i": i, "j": j, "u": float(pu[i, j]), "v": float(pv[i, j]),
                           "lat": float(plat[i, j]), "lon": float(plon[i, j])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"directions": records, "pixels": pixels}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
