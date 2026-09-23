# Architecture notes

Real-time GI: SDF geometry clipmap + radiance cascades, static/dynamic
occlusion split. PS4 hardware specs (1.84 TFLOPS, 176 GB/s) are used as
an illustrative performance budget, not a literal shipping target.
Prototyping in Bevy (Rust/WGPU); dev hardware is an Intel Iris Xe
(Tiger Lake GT2) — ~2.4 TFLOPS peak (ahead of PS4) but bandwidth-poor
shared system RAM (well under PS4's dedicated 176 GB/s), so treat
absolute frame-time numbers gathered on it as directional, not final.
No physics, minimal additional systems for now — pure renderer focus.

## Data structures
- **Geometry clipmap**: 3 tiers, toroidal, camera-centered. 12.5cm /
  50cm / 2m voxels.
- **Shadow clipmap**: 3 tiers, decoupled non-overlapping radial
  intervals. Marched every couple of frames, interpolated between
  marches — not every frame.
- **Probe radiance field**: high spatial density, 16-24 rays/probe,
  SH or octahedral atlas encoding.

## Static vs. dynamic split (the core design principle)
- **Static SDF march**: slice-triggered — only the newly-revealed
  toroidal slice marches when the camera crosses into it, not the
  whole field every frame. LOD by cascade tier (distance from
  camera), *not* object-binned — there's no discrete object list at
  this point, it's a merged continuous field.
- **Dynamic objects**: OBB + bin broad-phase culling, reused across
  both the shadow SDF march and re-evaluating occlusion at cached
  static hit points every frame (this term can't be slice-amortized —
  a moving occluder can change on any frame regardless of camera
  movement). Culls entire probes/directions via the same bins.
- Non-animated (rigid) dynamic objects are a candidate for a baked
  per-object SDF instead of a capsule proxy (Lumen's Mesh Distance
  Field approach) — capsule proxies stay for skinned/deforming
  objects, which can't be cheaply re-baked per frame.
- **On-screen vs. off-screen gather rays are NOT equally amortizable**:
  on-screen hits reproject into live `SceneColor` (changes every
  frame from dynamic lighting/actors) so they can't ride the slice
  cadence; off-screen hits query the static SDF + static shadow
  clipmap and can.

## Multi-bounce
Achieved via the cascade hierarchy converging within a single frame
(deterministic), not temporal accumulation across frames (Lumen-style,
stochastic). Deliberate trade: no propagation lag/ghosting, but the
cascade interval/seam math has to be right since there's no denoiser
smoothing over mistakes.

## Open problem: albedo source
A pure distance-only SDF gives hit position + normal, nothing else —
confirmed this is exactly the problem Lumen's Surface Cache solves.
Since lighting here is computed analytically (not cached/relit like
Lumen's cards), we only need the *material* half of what Surface
Cache does, not the lighting-cache half. Options, undecided:
1. Bake albedo into clipmap voxels (simplest, ties material res to
   12.5cm finest-tier geometry res — will blur fine detail)
2. Separate, independently-resolved albedo volume
3. Per-object flat/dominant color (cheap, coarse)
4. Full card-atlas replica of Lumen's approach (most accurate, most
   engineering)
Design principle either way: missing coverage should degrade as lost
energy, not leak.

## Why Bevy over Unity/Unreal/Godot
Building a from-scratch lighting pipeline needs a thin substrate to
build *on top of*, not an opinionated renderer to fight or carve
compute-shader access out of. Godot's RenderingDevice API is real but
framed as an advanced escape hatch (CompositorEffect hooks into an
existing Forward+ renderer); Bevy's equivalent is the normal way to
work. Bevy's own occlusion culling (two-phase GPU HZB, same technique
as UE5 Nanite) and LOD are real but marked experimental — this is
orthogonal to the GI work (governs the rasterization/G-buffer front
half, not the lighting back half) so instability there doesn't block
this project.

## Roadmap
1. Baseline: deferred G-buffer, depth pre-pass/Hi-Z, static shadow
   raster, clustered direct lighting. (Scaffold exists: main.rs)
2. 2D/screen-space radiance cascade testbed against a flat test scene,
   validated against existing public reference implementations,
   before touching the clipmap at all.
3. Integrate the SDF clipmap tiers, go 3D. This is the long pole —
   the 3D generalization of radiance cascades is still an open
   research problem (a foundational paper on this landed mid-2026).
4. Add the static/dynamic split and OBB culling.
5. Profile against PS4-derived ms budgets using GPU timestamp
   queries, on whatever hardware is in hand.
6. Stress-test the two known-unverified failure modes: fast camera
   movement crossing multiple slices at once (uncapped burst risk),
   and a hot bin with many overlapping dynamic OBBs (wavefront
   divergence concentrated in one dispatch).

## Prior art / precedent consulted
- UE5 Lumen: SDF/voxel tracing + Surface Cache, ~2 years reveal-to-
  ship, ~3-4 years to broad production trust — mostly due to
  denoising a stochastic technique across arbitrary content, dual
  hardware/software RT backends, and institutional trust lag, not
  algorithmic difficulty alone.
- Bevy Solari (jms55): tried screen-space probes then radiance
  cascades, hit real complexity/quality issues, a broken world-space
  radiance cache (later diagnosed as implementation error, not
  algorithmic), shelved after ~1 year. Later rebuilt around ReSTIR +
  hardware RT + world irradiance cache instead — which is *also*
  still fighting light leaks (cache cell spans a surface boundary)
  and denoiser correlation artifacts as of the most recent release.
  Lesson: even with a paper and reference implementation, the
  failure mode is almost never the algorithm — it's an unstated
  engineering decision (encoding format, cell-size-vs-ray-length
  relationship, coordinate space) that doesn't crash, just produces
  plausible-looking-wrong output.
