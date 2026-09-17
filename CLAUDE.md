# PlanetCreator — notes for Claude

- Python via `uv` (`uv run ...`, `uv sync --extra dev`). Tests: `uv run pytest`. Lint: `uv run ruff check src tests scripts`.
- Data lives in `data/` (gitignored). Raw downloads in `data/raw/<dataset>/`, baked cubes in `data/cube/<res>/`.
- `cubesphere.py` defines the face/uv/axis convention; the Rust/Godot runtime must match it. Change it only with a matching change on both sides and a test.
- Validation is by held-out regions (`bake.DEFAULT_HOLDOUT`), never random patches.
- Design docs in `docs/`; keep them updated when a decision changes.
