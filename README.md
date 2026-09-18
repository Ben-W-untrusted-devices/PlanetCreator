# PlanetCreator

Neural planetary terrain generator for a game. Trained on Earth (and other
solar-system bodies') elevation and satellite imagery; generates novel
planets — and Orbitals, Ringworlds, Halos — with similar local character but
new global layouts, controllable by game parameters.

- [docs/approaches.md](docs/approaches.md) — generation design and chosen pipeline
- [docs/tech-demo.md](docs/tech-demo.md) — real-time space-to-surface demo plan

Engine decision: **Godot 4 + Rust GDExtension** for the runtime; Python/PyTorch
for data and training. The cube-sphere convention in
`src/planetcreator/cubesphere.py` is the contract between the two.

## Setup

```bash
uv sync --extra dev
uv run pytest
```

## Data pipeline

```bash
uv run scripts/download_data.py --list          # show sources, sizes, licences
uv run scripts/download_data.py --months 200407 # ~1.6 GB into data/raw/
uv run scripts/build_cube.py --res 4096         # -> data/cube/4096/<face>/<layer>.npy
uv run scripts/preview_cube.py data/cube/4096   # PNGs in data/cube/4096/preview/
```

Baked layers per face (`data/cube/4096/<face>/`):

- direct: `height` (m, ETOPO), `rgb` (uint8, Blue Marble), `tavg`/`trange`/`prec`
  (WorldClim annual summaries, oceans filled with a latitude-band mean), `land`,
  `lat`/`lon`, `holdout` (validation regions: whole continents, `bake.DEFAULT_HOLDOUT`)
- derived from height (`planetcreator.hydro`, numba): `water` (ocean + lakes from
  priority-flood depressions and exactly-flat DEM regions), `flowacc`
  (log10 upstream area km², D8 routing on the global equirect grid),
  `dist_{up,down,left,right}` (log1p km to ocean along the face axes — a
  moisture-source / rain-shadow channel)
- `ctx_*`: the same fields at 1/8 resolution, computed from the 8× downsampled
  height, on a face padded by 896 px so every patch has a 2048-px context window

`planetcreator.dataset.CubePatchDataset` yields aligned patches plus their
context window, with rotation/flip augmentation that permutes the directional
channels accordingly (tested by recomputing distances on the rotated mask).

## Terrain model

`planetcreator.terrain.TerrainNet`: a UNet over the patch whose bottleneck also
sees the encoded context window (local crop + global pool). Two modes share
one input layout (`planetcreator.features`):

- `colour`: fine height + derived channels + climate → RGB
- `joint`: coarse height + derived channels (from the context) + climate →
  fine height (residual over the coarse) + RGB

```bash
uv run scripts/train_terrain.py --mode joint --steps 8000 --out runs/joint      # ~1 s/it on an M1
uv run scripts/train_terrain.py --mode joint --resume runs/joint/last.pt --adv 0.05 --steps 4000 --out runs/joint_adv
uv run scripts/render_face.py runs/joint/last.pt --face ny                     # whole-face prediction vs truth
```

Training writes `log.csv`, `val_<step>.png` (rows: input height, predicted
height, predicted RGB, true RGB, true height on held-out patches) and `last.pt`.

## Synthetic planets

```bash
uv run scripts/make_planet.py --seed 3 --checkpoint runs/joint/last.pt   # ~10 min on an M1
```

`planetcreator.procgen` builds a noise base map (continents, shelves, mountain
belts, continental basins) and climate; `planetcreator.erosion` runs
stream-power incision + uplift + diffusion on the equirect grid (fastscape
scheme, numba) to carve drainage networks and valleys; the eroded map is
box-downsampled to context scale and baked, and the joint model predicts the
fine height + RGB per face. Output goes to `data/planets/seed<N>/` in baked-cube
layout, so `PlanetMesh` loads it:

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path godot --resolution 1000x1000 -s scripts/capture.gd -- /tmp/p.png "$PWD/data/planets/seed3" 0.6 0.3
```

## Runtime (Godot 4 + Rust)

```bash
cd rust && cargo build          # needs rustc >= 1.94 (rustup stable); Homebrew's 1.90 is too old
cargo test -p planet-core       # includes the cross-check against Python cube-sphere vectors
```

Then open `godot/` in Godot 4.5+ (developed on 4.7.2 via `brew install --cask godot`)
and run `scenes/main.tscn`: the `PlanetMesh` node loads `data/cube/4096`
directly (baked `.npy` layers), builds six displaced, textured cube-sphere
faces, and the camera orbits it. No LOD yet.

Headless render to a PNG (used for checking without the editor):

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path godot --resolution 1280x800 -s scripts/capture.gd -- /tmp/earth.png
```

Known quirk: `godot --headless --import` crashes in 4.7.2 on macOS once the
extension is loaded (inside Godot's Metal shader converter); the windowed
editor and headless *runs* are fine. If a fresh checkout reports
`Cannot get class 'PlanetMesh'`, create `godot/.godot/extension_list.cfg`
containing `res://planetcreator.gdextension` or open the project in the
editor once.

- `rust/crates/planet-core` — engine-independent: cube-sphere convention
  (mirrors `cubesphere.py`; verified by `tests/data/cubesphere_vectors.json`,
  regenerate with `scripts/export_cubesphere_vectors.py`), `.npy` reader,
  baked-cube loader.
- `rust/crates/planet-godot` — the GDExtension (`planetcreator.gdextension`).
  Axis mapping planet-core → Godot is `(x, y, z) → (x, z, -y)`: north pole +Y.

## Layout

- `src/planetcreator/` — Python library (data, models, training)
- `scripts/` — data download / preprocessing / training entry points
- `tests/` — pytest
- `rust/` — Cargo workspace (core + GDExtension)
- `godot/` — Godot project
- `docs/` — design notes
- `data/`, `checkpoints/`, `runs/` — gitignored, populated by scripts
