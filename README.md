# PlanetCreator

Neural planetary terrain generator for a game. Trained on Earth (and other
solar-system bodies') elevation and satellite imagery; generates novel
planets with similar local character but new global layouts, controllable
by game parameters (sea level, tectonic activity, climate, age).

See [docs/approaches.md](docs/approaches.md) for the generation design and
[docs/tech-demo.md](docs/tech-demo.md) for the real-time space-to-surface demo
plan, including Orbitals / Ringworlds / Halos.

## Setup

```bash
uv sync --extra dev
```

## Layout

- `src/planetcreator/` — library code
- `scripts/` — data download / preprocessing / training entry points
- `docs/` — design notes
- `data/`, `checkpoints/`, `runs/` — gitignored, populated by scripts
