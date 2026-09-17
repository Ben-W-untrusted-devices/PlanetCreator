# Generating novel planets from one Earth: approaches

## Reframing the "one sample" problem

Earth is one *planet* but it is not one *sample*. At 1 km resolution the land
surface alone is ~150 M pixels; at 8 km tiles that is millions of overlapping
training patches, spanning every climate and geomorphology we'd want on a
game planet. What we genuinely have only one example of is the **macro
layout** (continent shapes, plate arrangement, ocean/land ratio). So:

- **Local/meso structure (< ~500 km)** — ridges, drainage networks, dune
  fields, coastlines, glaciation, and how they colour under a given climate —
  is *learnable* from Earth, with millions of samples.
- **Macro structure (> ~1000 km)** — where continents go — is *not* learnable
  from one planet and should come from a procedural/simulated source
  (tectonics or shaped noise) that we control with game-facing knobs.

Everything below follows from splitting the problem at that scale boundary.

Two extra data levers worth using:

1. **Other bodies have global DEM + imagery too**: Mars (MOLA + Viking/HRSC
   mosaics), the Moon (LOLA + LRO WAC), Venus (Magellan radar), Mercury
   (MESSENGER). For the *height* model these add genuinely different
   geomorphology (impact cratering, no fluvial erosion, volcanic plains).
2. **Earth contains all the climates, just arranged spatially.** If we
   condition patches on local temperature/precipitation (WorldClim), then at
   generation time we can set those planet-wide: Sahara patches → desert
   world, Antarctica → ice world, Amazon → jungle world.

## Candidate approaches

### 1. Wave Function Collapse (WFC)

WFC is a constraint solver over a *discrete* tile set with *local* adjacency
rules. To apply it to continuous height + RGB you'd quantise patches into a
few hundred tile types (k-means over 32×32 windows) and learn adjacency.

- Good: no training, fast, controllable, easy to add designer constraints.
- Bad: coherence is purely local. Drainage networks (water must flow downhill
  to the sea), continental shelves, mountain arcs — these are non-local and
  WFC produces "plausible everywhere, wrong overall". Sphere topology is
  awkward. Continuous data quantised into tiles shows seams.
- Verdict: **good for a discrete biome / feature-placement layer**, not for
  the heightmap itself. Note that "WFC with a learned tileset and a learned
  prior" is exactly a VQ-VAE + discrete-diffusion/transformer model (approach 4).

### 2. Patch-based texture synthesis (non-neural baseline)

Zhou et al. 2007, *Terrain Synthesis from Digital Elevation Models*: user
sketches ridgelines, algorithm fills with real DEM patches via graph-cut
seams. PatchMatch/image-analogies variants do similar.

- Good: photoreal by construction, cheap, strong baseline to beat.
- Bad: visibly repeats source patches; no semantic understanding, so it also
  struggles with drainage; no conditioning on climate unless hand-engineered.
- Verdict: worth implementing as the **first baseline** (a day's work) so we
  can measure whether the neural models actually improve on it.

### 3. Conditional generative models on patches (recommended core)

Train on Earth (+Mars) patches with dense conditioning channels. Two models:

**(a) Height refinement**: input = coarse height (e.g. 1/8 resolution) +
climate channels → output = full-resolution height. This is a cascaded
super-resolution diffusion model (Imagen-style), or a pix2pix-style GAN if
runtime speed matters. It learns geomorphology: where the coarse map says
"mountain range", it invents realistic ridges, valleys and drainage.

**(b) Colourisation**: input = height, slope, latitude, climate channels →
output = satellite RGB. This is a dense image-to-image problem with millions
of paired samples — the cleanest supervised task in the whole project and
where the satellite imagery pays off most.

- Good: leverages the data where it is abundant; climate conditioning gives
  planet-wide "style" knobs for free; the coarse-height input is exactly the
  seam to the procedural macro layer.
- Bad: training cost (diffusion at 256² is fine on an M1 for prototyping, but
  the real run wants a rented GPU); runtime cost for a diffusion model
  generating a whole planet (mitigate: GAN, distillation, or generate offline).

### 4. VQ-VAE tokens + transformer / discrete diffusion ("learned WFC")

Encode (height, RGB) patches to a discrete codebook, then model token grids
with a transformer or MaskGIT. It is the neural generalisation of WFC: tiles
and adjacency are both learned, and attention gives non-local coherence.

- Good: unified model, natural for infinite/tiled generation, tokens are
  cheap to store and stream in a game.
- Bad: more moving parts than approach 3; quality at high resolution needs a
  good decoder. Reasonable **phase 2** once approach 3 works.

### 5. Single-image generative models (SinGAN / SinDiffusion / ConSinGAN)

Designed for exactly "learn from one image" via a multi-scale pyramid. Could
be fed the whole globe.

- Good: cheap, produces varied internal remixes.
- Bad: outputs are recognisably "Earth shuffled"; poor control; no climate
  knobs. Interesting experiment, not the backbone.

### 6. Simulation for the macro layer (not learned — and that's fine)

- Tectonics: Cortial et al. 2019 *Procedural Tectonic Planets*, or a
  simplified plate-drift + collision/subduction model on a cube-sphere.
- Simple climate: latitude → insolation; altitude → lapse rate; prevailing
  winds + distance-to-ocean → moisture; gives temperature/precip maps that
  feed the conditioning of approaches 3/4.
- Hydraulic erosion (GPU): optional physical prior before or after the NN.

This layer is what makes planets *novel* rather than Earth remixes, and it
is where the game's parameters live (sea level, tectonic activity, age,
axial tilt, global temperature, moisture).

## Recommended pipeline

```
game params ─► tectonic/noise macro height (cube-sphere, ~128 px/face)
            ─► simple climate sim  ─► temp / precip / latitude channels
            ─► [NN a] height refinement (coarse ─► fine)   ← trained on Earth+Mars DEM patches
            ─► [NN b] colourisation (height+climate ─► RGB) ← trained on Blue Marble + ETOPO + WorldClim
            ─► (optional) WFC / rules for discrete features: cities, resources, biome tags
```

Sphere handling: generate on the six cube-sphere faces with overlapping
borders (MultiDiffusion-style blending) or with latitude as a conditioning
channel on an equirectangular grid; cube-sphere avoids polar distortion.

## Suggested order of work

1. **Data pipeline** — download, reproject to cube-sphere, cut aligned
   patches of (height, RGB, slope, latitude, temp, precip). Everything
   depends on this.
2. **Colourisation model (3b)** first: cleanest supervised task, immediate
   visual feedback, validates the pipeline. Feed it a noise/erosion heightmap
   and you already have a usable planet.
3. **Patch-synthesis baseline (2)** for height, to have something to beat.
4. **Height refinement model (3a)**, conditioned on coarse height + climate.
5. **Macro layer (6)** with game-facing knobs; wire the whole pipeline.
6. Evaluate; consider VQ/transformer (4) if tiling coherence is the weak
   point.

## Evaluation (so "looks realistic" is measurable)

- Hypsometric curve, slope histogram, and terrain power spectrum vs Earth.
- Drainage density / fraction of pixels that flow to an ocean.
- FID/KID on patches against held-out Earth regions (hold out whole
  continents, not random patches — random splits leak via overlap).
- Blind A/B against the patch-synthesis baseline.

## Data sources

| Layer | Source | Resolution |
|---|---|---|
| Global topo + bathymetry | ETOPO 2022 / GEBCO 2024 | 15 arc-sec (~450 m) |
| Land DEM (high-res) | Copernicus GLO-30, SRTM | 30 m |
| Cloud-free RGB | NASA Blue Marble Next Generation (monthly) | 500 m |
| Land cover | MODIS MCD12Q1, ESA WorldCover | 500 m / 10 m |
| Climate | WorldClim 2.1 (temp, precip) | ~1 km |
| Mars | MOLA DEM + Viking MDIM / HRSC mosaic | 463 m / 232 m |
| Moon | LOLA DEM + LRO WAC mosaic | 118 m / 100 m |

Note: Blue Marble is shadow-minimised, which matters — raw satellite RGB
bakes in sun direction, which the model would then learn as a fixed
"north-west lit" texture.
