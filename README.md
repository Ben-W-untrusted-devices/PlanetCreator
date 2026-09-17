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

## Layout

- `src/planetcreator/` — library code
- `scripts/` — data download / preprocessing / training entry points
- `tests/` — pytest
- `docs/` — design notes
- `data/`, `checkpoints/`, `runs/` — gitignored, populated by scripts
