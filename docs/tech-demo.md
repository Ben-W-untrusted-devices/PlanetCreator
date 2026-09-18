# Tech demo: space → surface in real time, planets and megastructures

Goal: start in orbit, fly down, walk around, in real time. Also generate
Banks Orbitals, Niven Ringworlds and Halos, not just spheres.

## Core rule

A planet at 1 m is ~5e14 pixels. Nothing generates that; every
space-to-surface engine is a lazy hierarchy. So generation must be
**hierarchical, lazy, deterministic per tile, and cheap per level**.

| Scale | Source | Notes |
|---|---|---|
| Planet → ~1 km | Baked NN output (macro → refinement → colour + land-cover) | Cube-sphere 6×4096², ~600 MB. Offline. |
| ~1 km → ~30 m | Neural LOD: tiny 2–4× per-level tile upsampler (~1M params, GAN/distilled), on-GPU | Training data exists here (Copernicus 30 m, Sentinel-2 10 m). Seed = tile id. Padded context for seams. |
| < 30 m | Procedural + assets | No global sub-10 m DEM to learn from. Noise displacement, material splat by land-cover + slope, scatter. |

The colourisation net gets a **land-cover class head** (ESA WorldCover /
MODIS). The renderer uses it for materials, vegetation scatter, and to
condition procedural detail.

**v1 demo skips neural LOD**: bake at ~1 km, procedural below. Neural LOD
is v2 — it is the riskiest runtime component.

## Engine requirements (non-ML, unavoidable)

- Cube-sphere quadtree, CDLOD-style morphing, background tile generation
  ahead of the camera.
- Floating origin + double-precision positions (Godot: `precision=double`).
- Logarithmic / reversed-Z depth.
- Precomputed atmospheric scattering (Bruneton) — the descent shot depends
  on it.
- Runtime inference: small conv upsamplers as compute shaders (Metal/WGSL);
  no ML runtime in the demo binary. Keep models small enough for this.

Engine choice: **Godot 4 (Metal) + Rust GDExtension** for terrain/LOD/
inference. Free UI, character controller, physics, double-precision build.
Alternative: Bevy/wgpu, fully custom.

## Megastructures

The patch models never see topology (coarse height + climate → fine height
+ colour), so a megastructure is a different **surface parameterisation**
plus a different **macro layer**.

### Surface abstraction

```
trait Surface {
    fn position(u, v, level) -> DVec3;   // world position
    fn up(u, v) -> Vec3;                 // gravity is -up
    fn tile_neighbours(tile) -> [...];   // wraps in θ for rings
}
```
Sphere = cube-map faces. Ring/Orbital/Halo = cylinder band, periodic in θ,
bounded by rim walls. No polar distortion — easier than the sphere.

### Macro layer: designed, not simulated

Built worlds have no tectonics; their makers laid out seas, ranges, rim
walls. This is where **WFC fits**: a discrete designer-intent layer
(sea / plain / range / lake district / rim foothills) with adjacency rules
and optional hand-placed constraints, rasterised to coarse height. The NN
then adds real-looking geomorphology. Artificial world, real erosion.

### Climate

Uniform insolation across the band (no latitude); day/night from spin tilt
(Orbital) or shadow squares (Ringworld). Same temp/precip channels out.

### Scale

| Structure | Size | Strategy |
|---|---|---|
| Halo | 10,000 km ⌀, ~320 km wide, ~3e6 km² | Bake fully. First target. |
| Orbital | 3M km ⌀, ~6,000 km wide, ~6e13 km² | Macro generated lazily / repeating-varied; requires neural LOD path. |
| Ringworld | 1 AU radius, 1.6M km wide, ~3M Earths | Everything lazy; generate a corridor around the player only. |

### The arch

Far side of the ring rising overhead through atmosphere. Needs scattering
along the arch, atmosphere as a slab held by rim walls, no fog cheating,
doubles. Camera up-vector from `Surface::up`; "down" is radially outward.

## Runtime skeleton (done)

`rust/crates/planet-core` (cube-sphere + npy + baked-cube loader, checked
against Python) and `rust/crates/planet-godot` (`PlanetMesh` node: six
displaced, textured faces from a baked cube, no LOD). `godot/scenes/main.tscn`
shows the baked Earth with an orbit camera. Single-precision Godot for now;
the double-precision build comes with the floating-origin work in phase 2.

## Phases

1. Data pipeline + colourisation/land-cover net. Deliverable: procedural
   heightmap → Earth-like textured sphere in a viewer.
2. Baked planet, real-time descent: quadtree, floating origin, log depth,
   scattering, procedural sub-km detail, walkable. Standalone demo.
3. Halo: cylinder-band Surface, WFC/designed macro, rim walls, arch.
4. Neural LOD (1 km → 30 m upsampler as compute shader). Unlocks Orbital.
5. Height refinement net for the baked base; vegetation/water/clouds.

## Risks

- Phase 2 is the largest chunk of work and is not ML.
- Neural LOD at 60 fps on an M1 is plausible, unproven until measured.
- Ground-level fidelity is bounded by procedural detail, not the NN.
