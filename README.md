# PlanetCreator

Neural planetary terrain generator for a game. Trained on Earth (and other
solar-system bodies') elevation and satellite imagery; generates novel
planets — and Orbitals, Ringworlds, Halos — with similar local character but
new global layouts, controllable by game parameters.

- [docs/approaches.md](docs/approaches.md) — generation design and chosen pipeline
- [docs/tech-demo.md](docs/tech-demo.md) — real-time space-to-surface demo plan
- [docs/plan-resolution.md](docs/plan-resolution.md) — resolution stages (2.4 km → 300 m → 37 m → 5 m), grid convention, current-stage fixes

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
- `joint`: coarse height + derived channels (from the context) + climate +
  deterministic per-pixel noise → fine height + RGB. The height is a residual
  over a smooth mean-preserving upsample of the coarse height, projected so that
  every 8×8 block of the output averages exactly to its coarse texel (the coarse
  level is always the block mean of the fine one; no steps at block edges).

Losses: L1 + gradient L1 on rgb and height; optional VGG16 perceptual term
(`--perc`), optional conditional PatchGAN (`--adv`) with an R1 gradient penalty
(`--r1`). Validation is on held-out continents (South America, Australia/NZ).

### Training

Training takes hours on the M1 (1–3 s/it depending on model size and losses),
so always run it detached from the terminal or Claude session that starts it,
with the machine kept awake:

```bash
scripts/train_v2.sh                       # the current recipe: two phases, ~12 h on an M1, self-resuming
tail -f runs/joint4/train.log             # watch (phase 2 logs to runs/joint4_adv/train.log)
kill $(cat runs/train_v2.pid)             # stop
scripts/train_v2.sh                       # run again after any interruption: continues from the last checkpoint
```

For a single custom run use the generic launcher, which wraps
`train_terrain.py` in `nohup` + `caffeinate` and writes a pid file:

```bash
scripts/train_bg.sh runs/myrun --mode joint --base 48 --perc 0.1 --steps 10000
scripts/train_bg.sh runs/myrun --continue runs/myrun/last.pt --mode joint --base 48 --perc 0.1 --steps 10000   # resume
```

`--continue` restores weights, optimiser, LR schedule and step (the dataset is
deterministic per index, so it resumes on the exact batch it would have seen
next); `--resume` takes only the weights (used to start an adversarial phase
from an L1 checkpoint). Runs survive the launching shell exiting; what they do
not survive is the laptop sleeping — `caffeinate -i` prevents idle sleep but not
a closed lid, so leave it open or on an external display.

Each run directory gets `log.csv`, `val_<step>.png` (rows: input height,
predicted height, predicted RGB, true RGB, true height on held-out patches),
`last.pt` and `step_<n>.pt` (GAN phases oscillate; keep every checkpoint).

```bash
uv run scripts/render_face.py runs/joint4/last.pt --face ny   # whole-face prediction vs truth
```

Judge height outputs with a 1:1 hillshade, not a downsampled height image —
block and grid artefacts are invisible in the latter (`make_planet` writes
`preview/<face>_hillshade.png`).

## Released model

`models/joint_latest.pt` is the current 2.4 km-stage joint model (fp16 weights,
17 MB), tracked in the repo so a fresh clone can generate planets without the
hours of training. `models/joint_latest.json` records its training step,
validation metrics and git commit. Export a new one with
`scripts/export_release.py runs/<run>/last.pt` — only after checking a
hillshade and a held-out render; the export is a manual gate.

## Synthetic planets

```bash
uv run scripts/make_planet.py --seed 3        # uses models/joint_latest.pt; ~15 min on an M1
```

`planetcreator.procgen` builds a noise base map (continents, shelves, mountain
belts, continental basins); `planetcreator.erosion` runs stream-power incision
+ uplift + diffusion on the equirect grid (fastscape scheme, numba, latitude-
aware spacing, stochastic routing + erodibility field) to carve drainage
networks; the eroded map is smoothed to Earth-like coarse roughness, climate is
derived from that smoothed map, and it is baked to context scale with the fine
inputs padded past each face edge. The joint model then predicts fine height +
RGB per face and the padded outputs are cross-faded across face edges. Output
goes to `data/planets/seed<N>/` in baked-cube layout, so the Godot nodes load it:

```bash
/Applications/Godot.app/Contents/MacOS/Godot --path godot --resolution 1000x1000 -s scripts/capture.gd -- /tmp/p.png "$PWD/data/planets/seed3" 0.6 0.3
```

## Runtime (Godot 4 + Rust)

```bash
cd rust && cargo build          # needs rustc >= 1.94 (rustup stable); Homebrew's 1.90 is too old
cargo test -p planet-core       # includes the cross-check against Python cube-sphere vectors
```

Then open `godot/` in Godot 4.5+ (developed on 4.7.2 via `brew install --cask godot`)
and press F5. The main scene is `scenes/descent.tscn`: a `PlanetLod` node
(cube-sphere quadtree, floating origin, fractal detail below the baked data,
padded seamless face textures), a single-scattering atmosphere post-process
(`shaders/atmosphere.gdshader`, sky + aerial perspective from one shader), a
fly camera and a HUD. It starts 15,000 km above a generated planet
(`data/planets/seed3`); set the Planet node's `cube_dir` to
`res://../data/cube/4096` for Earth.

Controls: right-drag or Tab to look, WASD to move, Q/E down/up, Shift x10,
Ctrl /10, F to face the planet, Esc releases the mouse. Speed scales with
altitude; the camera starts aimed at the planet and the HUD says where it is.

`scenes/main.tscn` is the older orbit view (`PlanetMesh`, no LOD).

Scripted screenshots: `scripts/capture_descent.gd -- out.png cube_dir lat lon alt_m yaw pitch [frames] [debug_mode]`.

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
- `models/` — released weights (tracked, fp16) with provenance JSON
- `data/`, `runs/` — gitignored, populated by scripts (raw downloads, baked cubes, planets, training runs)
