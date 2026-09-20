# PlanetCreator — notes for Claude

- Python via `uv` (`uv run ...`, `uv sync --extra dev`). Tests: `uv run pytest`. Lint: `uv run ruff check src tests scripts`.
- Data lives in `data/` (gitignored). Raw downloads in `data/raw/<dataset>/`, baked cubes in `data/cube/<res>/`.
- `cubesphere.py` defines the face/uv/axis convention; `rust/crates/planet-core/src/cubesphere.rs` mirrors it and is checked against `tests/data/cubesphere_vectors.json` (regenerate with `scripts/export_cubesphere_vectors.py`). Change both together.
- `planetcreator/features.py` defines the model conditioning channels (per-pixel `COND_CHANNELS` and context `CTX_CHANNELS`); the runtime must build the same tensors. Slope is metres per *training pixel* (`TRAIN_PX_KM` ≈ 2.45 km), so pass the raster's `px_km` to `build_cond`/`infer_face` for any resolution other than a 4096 face.
- Derived channels (`water`, `flowacc`, `dist_*`, `ctx_*`) come from `planetcreator.hydro` + `bake.py` and must be computed identically for Earth and synthetic planets (both go equirect → `bake`).
- Rotation/flip augmentation must permute `dist_*` channels (`dataset.augment_layers`); keep `tests/test_dataset_aug.py` passing if you touch it.
- Rust: build with rustup's stable (>= 1.94), e.g. `PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" cargo build` from `rust/`; Homebrew's cargo is too old. Run `cargo clippy --all-targets` and `cargo fmt`.
- Godot project is `godot/`; the extension manifest points at `rust/target/{debug,release}`.
- Validation is by held-out regions (`bake.DEFAULT_HOLDOUT`), never random patches.
- Design docs in `docs/`; keep them updated when a decision changes.
- Training runs go through `scripts/train_bg.sh` / `scripts/train_v2.sh` (detached, `caffeinate`, resumable), never a foreground Bash call. Judge height outputs by 1:1 hillshade. Exporting to `models/` is a manual gate after review.
