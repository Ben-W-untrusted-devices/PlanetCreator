# Plan: resolution stages and the current-stage fixes

Gate: a stage is only started when the previous one is signed off. No training
run for stage A until the 2.4 km stage is complete.

## Grid convention: cell-centred averages, not vertex samples

Every level is a **cell-centred** raster: a texel is the average over its cell,
and the coarse level is the block average (8×8) of the fine one. This is what
`bake.box_downsample` does for Earth's context layers, what
`features.coarse_from_ctx` assumes when it upsamples (`align_corners=False`
treats texels as cells, so the coarse texel centre sits at fine coordinate
8i + 3.5), and what the generated planet's coarse map is (an average of the
eroded map). The runtime samples any level bilinearly at mesh vertices
(`CubeHeight::sample_dir` puts texel centres at (x + 0.5) / n), so vertices
never need to coincide with texel centres.

Why not vertex samples (coarse texel = the fine value at every 8th vertex)?
Point sampling aliases: the coarse input would carry whatever happened to be
at those points rather than the region's content, and the network would learn
to invent detail from an aliased signal. A coarse sensor measures an average,
and so does our generator, so averages are the honest training input.

What a vertex-nested scheme *would* buy is exact agreement between levels at
shared points. We get the equivalent for cell-centred grids by making the
refinement **mean-preserving**: the network's residual is forced to have zero
mean over every 8×8 block, so the coarse level is exactly the average of the
fine level (a Laplacian-pyramid constraint). Then switching LOD level changes
detail but never the local mean, and the coarse map is always the truth about
the fine one. This is part of the current-stage fixes below.

## Current stage (2.4 km): fixes before sign-off

Observed in the descent (screenshot 2026-09-19):

1. **Content seam at cube-face edges.** The joint model is run per face; tiles
   on either side of an edge hallucinate different detail.
   Fix: bake the fine input layers with a padded margin, run inference on the
   padded face, and cross-fade each face's padded output with the neighbour's
   interior over the margin. Both faces converge to the same values at the
   edge.
2. **Blocky coasts, lakes and land patterns.** Coarse conditioning channels
   are nearest-upsampled 8×, and the water mask in particular becomes 8-px
   staircases the model reproduces.
   Fix: store the coarse water channel as a coverage fraction and upsample all
   coarse channels bilinearly. Evaluate on Earth's held-out continents without
   retraining first (the model saw hard blocks in training; a soft ramp is a
   small shift), then retrain with it.
3. **Diagonal hatching / streaking.** Partly the adversarial fine-tune's stripe
   artefact, partly grazing-angle texture stretching in the renderer.
   Fix: anisotropic filtering on the terrain material (renderer); lower
   adversarial weight and a perceptual term in the retrain (model).
4. **Mean-preserving residual** (above) so the coarse map and the fine map
   agree, and LOD transitions do not pop.
5. **Deterministic refinement**: a per-tile seed channel so a quadtree node
   always refines to the same detail (needed for the runtime later; costs one
   input channel, so it goes into the same retrain).

Fixes 1–3 (renderer part) need no training. Fixes 2 (retrain part), 3 (model
part), 4 and 5 are one retraining run of the existing model (~4 h on the M1).

## Stage A: 2.4 km → 300 m

Same `TerrainNet`, trained on real pairs at those scales.

- Height truth: Copernicus GLO-90 (90 m, free) block-averaged to 300 m; the
  coarse input is the same averaged to 2.4 km.
- Colour truth: Sentinel-2 cloudless mosaic (EOX, free, tiled) at ~300 m.
- Volume: ~10–15 GB (a few GB each globally at these scales).
- Runtime: ONNX export; `ort` (CoreML on macOS) inference inside `PlanetLod`
  on a worker thread; the quadtree asks for the 300 m level as you descend;
  fractal detail remains the fallback below the deepest network level.

## Stage B: 300 m → 37 m

- Height: Copernicus GLO-30 (30 m). Colour: Sentinel-2 at 10 m averaged to 37 m.
- ~2,000 sampled 1° tiles across biomes, ~30–60 GB. Global would be ~1 TB; not needed.
- 30 m is the floor for global height data.

## Stage C: 37 m → ~5 m (appearance only)

- No global fine DEM exists; synthesise colour + normal detail from NAIP 1 m
  aerial imagery (USDA, public domain), ~30–100 GB of sampled quarter-quads.

## Disk

Main drive has ~90 GB free. Stage A fits easily; stage B needs the raw tiles
deleted after baking, or an external drive. Nothing under `data/`, `runs/` is
ever committed; small test fixtures under `rust/**/tests/data` are.

## Training recipe notes (2026-09-21)

- Recipe v2 phase 1 (L1 + gradient + VGG perceptual, base 48, 10k steps):
  held-out height 22.9 m, RGB 0.063 — best so far.
- R1 gradient penalty on the PatchGAN: dropped. The penalty must be normalised
  per logit (a 30x30 logit map otherwise scales it ~900x), and even then any
  gamma strong enough to matter stops the discriminator learning within
  thousands of steps; the unregularised discriminator needs input-gradient norms
  of ~50 to separate real from fake. Phase 2 uses plain hinge GAN at 0.05 with
  the perceptual term kept on to anchor colour.
