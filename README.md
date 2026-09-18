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

Baked layers per face: `height` (m, ETOPO), `rgb` (uint8, Blue Marble),
`tavg`/`trange`/`prec` (WorldClim annual summaries, oceans filled with the
latitude-band mean), `land` (WorldClim valid mask), `lat`/`lon` (deg),
`holdout` (validation regions: whole continents, see `bake.DEFAULT_HOLDOUT`).

`planetcreator.dataset.CubePatchDataset` yields random aligned patches with
rotation/flip augmentation, deterministic per `(seed, index)`.

## Colouriser (phase 1 model)

`(height, slope, ocean, lat, tavg, trange, prec) -> rgb`, a 7.9M-param UNet
(`planetcreator.colorize`). Conditioning channels and their scales are in
`planetcreator.features` — the runtime must build the same tensor.

```bash
uv run scripts/train_colorize.py --steps 20000 --out runs/colorize   # ~0.7 s/it on an M1
uv run scripts/colorize_face.py runs/colorize/last.pt --face ny       # full-face render vs truth
```

Training writes `log.csv`, `val_<step>.png` (rows: height / prediction /
truth on held-out patches) and `last.pt` to the run directory.

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
