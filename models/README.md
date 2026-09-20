# Released models

Weights exported by `scripts/export_release.py` (fp16, weights only). Each `<name>.pt`
has a `<name>.json` with provenance (training step, validation metrics, git commit)
and a `<name>_preview.png` validation grid.

- `joint_latest.pt` — the 2.4 km-stage joint model (coarse height + climate + context →
  fine height + RGB). `scripts/make_planet.py` and `scripts/render_face.py` use it by default.

Loading fp16 weights into the fp32 model is handled by `TerrainNet.load`.
