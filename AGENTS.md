# Working notes

Handoff for whoever picks this up next. Delete freely — the git history carries
the reasoning. Kept short on purpose; ARCHITECTURE.md and EXPLAINED.md were
drawn up by the same people who let old handoff notes drift. AGENTS.md is read
and (usually) updated by the assistant each session, so keep asking "is this
still true?".

## Workstyle

**Never delete machinery to turn it off.** When the user asks to disable
something, comment it out and write a comment explaining why it is off and
what would put it back. The git history is a one-way rescue: a committed diff
survives, but anything deleted locally with the working tree uncommitted is
gone. That bit us with `stencil_touches_solid`, the caps, and the probe lift —
all had to be re-derived. Same for alternative implementations: keep the
previous one commented next to the new one with the measurement that decided
it, and delete freely once it is committed.

## What this is

Bevy 0.19 indirect lighting over a dual-channel heightmap (floor + ceiling per
column, `SPLIT = 0.0`). Currently baking `assets/models/room.glb` once at
startup into textures the cascade probes then trace against.

> **Scene note:** most of the historic notes below (leaks, counts, thresholds)
> predate the `room.glb` swap and referenced Sponza + `SPLIT = 5.0`. The
> numbers in them are stale; prefers `room.glb`. What still stands is the
> documented behaviour of each measurement, notably how winding-vs-normal-
> vs-trace tests behave and why the box culling modes fail.

Bevy's default ambient is **off**: `main.rs` inserts
`GlobalAmbientLight::NONE` (brightness 0) after `DefaultPlugins`, because the
room is lit by the baked cascade probes plus the sun and the stock brightness-80
flat ambient washed it out. Removing the line restores Bevy's default. Measured
at the wall camera (static sun): luminance 70.29 -> 54.71, mean |d| 14.45.

## Commands

```sh
cargo build --bin s_rc
cargo clippy --bin s_rc

# Legacy leak camera from the Sponza era:
BEVY_ASSET_ROOT=$PWD S_RC_CAM="-10.58,4.86,-3.21,-1.17,8.18,-3.83" \
  S_RC_INDIRECT=1 S_RC_OVERLAY=0 ./target/debug/s_rc
# Second storey:
#   S_RC_CAM="-9.20,4.88,-6.88,-0.63,6.62,-4.79"
# S_RC_PROBES=1          show the probe grid (instanced spheres, see State)
# S_RC_OVERLAY=1         keep the heightmap debug planes
# S_RC_CROSS_LAYER=1     cross-layer cascade merge (L key)
# S_RC_BILATERAL=1       cross-bilateral shade gather (B key)
# S_RC_NORMAL_BIAS=0     read the probe grid at the shading point + probe_visible
#                        (default is on: fetch at p + n*0.6 + V*0, trace skipped)
# S_RC_LOWPASS=0         no spatial low-pass in blend_sh (plain cross-fade, F key);
#                        the neighbour height gate is unconditional
# S_RC_BSPLINE_RECON=1   cubic B-spline shade reconstruction (C key), default off
# S_RC_SCREENSHOT=/tmp/x.png    also exits at frame 240
# Dump any baked image whole, to poke at it in numpy:
#   S_RC_DUMP_HEIGHT=/tmp/h.bin     Rg32Float  512x512x2, **x = ceiling, y = floor**
#   S_RC_DUMP_META_MIN=/tmp/m.bin   Rgba32Float, xyz = ceiling normal, true world-space
#   S_RC_DUMP_META_MAX=/tmp/m.bin   Rgba32Float, xyz = floor normal, true world-space
#   S_RC_DUMP_WINDING=/tmp/w.bin    R32Float, the winding slice at [`SPLIT`]
```

Dumps are row-major, so numpy indexes them `[z][x]`, the opposite of the shader's
`textureLoad(tex, vec2(x, y))`. Read the wrong way round, the open-sky count
comes out 0% and every count looks like noise.

## Gotchas that have cost real time

- **A failing shader pipeline does not stop the app.** Bevy logs
  `failed to process shader error` and carries on, so a green
  `heightmap bake verified` line does *not* mean the shading shaders compiled.
  Always finish a run with:
  `... 2>&1 | grep -icE "shader error|panicked|assertion"` and require `0`.
- **WGSL has no `pub`.** Exported functions are plain `fn`.
- **`create_empty_image`'s `channels` is a component count, not a byte count.**
  `R32Float` → 1, `Rg32Float` → 2, `Rgba32Float` → 4. Getting it wrong panics
  inside `bevy_image` with a bare `left/right` byte count.
- Don't trust a green build as evidence a runtime path works. Verify by
  measuring, and when verifying a shader change, force the input to a known
  value and confirm the output moves.
- **`cargo clippy` does not rebuild the binary.** Chaining it in front of a run
  measures the *previous* build; editing a Rust `const` and then verifying with
  clippy alone silently measures the old value. Use `cargo build` before a run
  that is supposed to reflect a Rust change.
- A run is reproducible: two runs of one binary are bit-identical. Cross-binary
  shots are comparable only when the bake is verified in both logs
  (`heightmap bake verified`), because the first bake can land before its
  pipeline exists — it logs `attempt 1 was empty; retrying` and then passes.
  `S_RC_PROBES=1` also changes pixels (it draws the instanced probe overlay),
  so keep it off for A/B shots.

## State

Probes sit in **two real layers per XZ cell** (`PROBES_PER_CELL = 2`, low bit of
the probe index = layer): a floor-layer probe a constant `PROBE_LAYER_GAP`
(0.1 m) above the column's stored floor surface, a ceiling-layer probe the same
distance below its stored ceiling. Columns without a usable interval (open sky,
or ceiling/floor missing) fall back to `split ± GAP`. `probe_position(c, probe,
layer)` / `probe_position_baked(...)` return these; the cascade index is
`cell_id * 2 + layer` in every buffer and dispatch (`layout::LAYERS = 2`,
`rays()`, `probe_base`, `rays_base`, `probe_words`, the `probe_positions` and
`gather` dispatches all ×2; `march_hits`, `cascades2_5d`, `gather_cascades` all
derive layer from `id % 2`).

**Both parallax corrections are gone.** The shade is a trilinear read of the two
layers (4 XZ neighbours × floor + ceiling, each layer gathered bilinearly and
renormalised, faded by how far the fetch anchor (see the normal-biased fetch
paragraph below; `p` by default) sits between the interpolated layer heights);
the SH is evaluated in closed form against the surface normal
(`sh_irradiance`), replacing the 32-sample Monte-Carlo `probe_irradiance` (still
commented in `shade_cascades.wgsl`). The cascade march/merge endpoint parallax
is off by default (`ParallaxState` default `false`), gated behind
`uniforms.parallax_endpoint != 0u` for A/B. The old single-layer bilinear read
survives behind `view_info.trilinear_enabled` (T key / `S_RC_TRILINEAR=0`) as a
reference. Measured at the interior camera: trilinear mean luminance 64.7 vs
bilinear 62.1, mean |d| 3.88, 31% of pixels past 0.05 digit.

**Cascade merges are per-layer, unless `cross_layer` is on.** Every parent-side
term (parent probe position, its stored segment geometry, its ray block) goes
through `parent_layer(layer)` in `heightmap_trace.wgsl`, which is the identity
by default and flips to `1 - layer` when `uniforms.cross_layer != 0u` — one
crossing per child, no oscillation up the chain. The L key /
`S_RC_CROSS_LAYER=1` toggles it; it is a march-time input, so `dispatch_cascades`
re-runs the one-shot march when it flips (the `marched_cross_layer` Local,
alongside `marched_parallax`). Measured at the interior camera: mean luminance
64.50 vs 64.70 separated, mean |d| 1.17, 2.8% of pixels past 0.05 digit.

**The shade's gather can also be a cross-bilateral (B / `S_RC_BILATERAL=1`, off
by default).** The B key keeps the trilinear footprint — the pixel's 4 corner
cells x 2 layers = **8 probes** — and replaces the fixed bilinear weights and
`height_agreement` fade with guide-driven weights, each tap weighted by the
bilinear tent, a normal kernel `max(0, n_p . n_q)^gamma` with `n_p` the
deferred G-buffer normal and `n_q` the **probe's own normal, assigned by its
bake**: the heightmap normal of its column (ceiling layer ->
`ceiling_normal`, floor layer -> `floor_normal`), or, when the probe sat
inside a wall and was moved sideways to the nearest open column
(`escape_texel_xz`), the direction of that move. A third rule went in and
came back out: probes whose burial sphere touched a wall but whose own
column was open facing the nearest solid texel — it lit 359/2706 columns
with horizontal normals but stepped where the rule boundary crossed a probe
cell and read as **shadow strips along structure edges**, so it is reverted.
Measured on `room.glb`: the escape branch fires on only **5** of 2706
cascade-0 probes (the ones standing on the central raised platform, whose own
column is solid); the perimeter walls almost never nudge because their probe
columns are interior/open. The normals are baked once
into `probe_positions`, which now holds **two vec4s per probe-with-layer**
(even = position, odd = normal; `probe_position`/`probe_normal` read `2*i` /
`2*i+1`, buffer sized `probes * 2` in `prepare_buffers`). The level and a
plane kernel `exp(-|n_p . (X_q - X_p)| / 0.5m)` with both positions in world
space — `X_p` reconstructed from prepass depth and `X_q` the stored probe
position. A wider footprint went in and came back out: RC is noiseless, so
the field between the neighbours is already a linear interpolation of them
and a (2R+1)^2 x 2 block had nothing extra to average — it only made the
pass 3x slower (50 taps -> 8 FPS). Kernels live as `BI_*` consts in
`shade_cascades.wgsl` (GAMMA 32, PLANE 0.5 m; there is no radius or spatial
sigma). The range kernels do most of the work: measured at the interior
camera with correct baked normals, the bilateral reads ~62.9 mean luminance
against the trilinear 62.2 with `GAMMA = 0` (the flat floor band unchanged,
mean|d| ~1.4) and ~54-56 against 62.8 with `GAMMA = 32` (walls and floor
darkened 5-9 digits, 40-50% of pixels past 0.05 — the orientation gate doing
its edge-preservation job; it cuts hard with no wide spatial support to
soften it). Note the per-tap `normalize(textureLoad(...))` read it replaced
produced **NaNs** on columns that recorded no surface (0/0, so the
`dot < 0.5` fallback never fired and the tap was silently dropped); the baked
path checks `dot(n,n) < 0.5` before normalizing, so the fallback works. The
default (off) build is bit-identical to the pre-bilateral frame.

**Bug 4: the bilateral normal gate collapsed on walls.** A vertical wall's
G-buffer normal is horizontal while every standard floor/ceiling probe's baked
normal is vertical, so `dot(n, n_q) = 0` and `pow(max(0, dot), BI_GAMMA)` = 0
for all 8 taps: the wall's `w_bi` collapsed onto only the nudged probes whose
horizontal normal happened to align, reading as dark columns. Fix (in the
bilateral tap loop): on a vertical pixel with a vertical probe normal
(`abs(n.y) < 0.5 && abs(n_q.y) > 0.8`) the hard gate is replaced by the soft
wrap `saturate(dot * 0.5 + 0.5)` (`dot = 0` -> 0.5); every other surface keeps
the `pow` gate that preserves edges. The previous line is kept commented
directly above. Measured at the wall camera with `S_RC_BILATERAL=1` and the
normal bias on (static sun): before the fix the bilateral sat at luminance
67.16 against the trilinear 70.29 (mean |d| 7.74); after, 72.98 (before-vs-after
mean |d| 5.85, 38.5% of pixels past 0.05, md5 `9e3adaee...` -> `cc85c59d...`).
The change is confined to the vertical band: the top 150 and bottom 150 rows
(ceiling/sky and floor) are bit-identical, so the wrap is not touching
floor/ceiling pixels.

**The shade's fetch can be normal-biased (N / `S_RC_NORMAL_BIAS`, on by
default).** Instead of reading the probe grid at `p`, `shade_cascades.wgsl`
reads it at `P_sample = p + n * BIAS_NORMAL + V * BIAS_VIEW` (`n` the G-buffer
normal, `V` the unit surface->eye direction, `BIAS_NORMAL = 0.6`, `BIAS_VIEW =
0.0`). Pushing the anchor along the normal moves it into the open half-space the
surface faces, so the cell selected is the one in front of the surface; that is
what `probe_visible` did with a trace, so the two are alternatives: the toggle
ON skips `probe_visible`, OFF (`S_RC_NORMAL_BIAS=0`) keeps it and reads at `p`.
The anchor drives both the XZ cell lookup (`gu`/`frac`) and the trilinear
vertical weight `t`. The `BIAS_NORMAL` start was the 0.4 m guard band
`mytodo.md` section 2 asks for (raised to 0.6 in the tree) and is the knob to
tune for leakage; note it also shifts `t` on
near-horizontal surfaces, so a floor pixel picks up a little ceiling-layer
light (about `BIAS_NORMAL` / room height). `BIAS_VIEW` is a second, off-by-
default slide toward the eye; `mytodo.md` section 4.3 is the rationale.

Measured at the wall camera (`S_RC_CAM=4.14,-0.96,2.53,0.81,0.03,-6.84`,
`S_RC_SUN=62,54.7 S_RC_SUN_STILL=1 S_RC_INDIRECT=1 S_RC_OVERLAY=0`), bias on vs
`S_RC_NORMAL_BIAS=0`: scene mean luminance 70.29 vs 68.98, mean |d| 2.78, 78% of
pixels past 0.05. `BIAS_NORMAL` 0.6 vs 0.2 changes the frame (md5 `f3bf7b28...`
vs `0af2202e...`), so the value reaches the shader. Both modes compile with 0
shader errors. The off-mode frame (`022a2588...`) is the tree's own default with
`probe_visible` restored — the working tree had it commented out and
`height_agreement` removed from the trilinear read by hand; those two edits are
kept (the legacy reference branch still uses both).

**Axes-aligned walls are γ-invariant by construction.** At the wall-face
camera (`S_RC_CAM=4.14,-0.96,2.53,0.81,0.03,-6.84`, `S_RC_SUN=62,54.7`) the
wall tile's delta vs the trilinear reference is flat across a γ sweep (71.1,
71.2, 71.3, 71.3 for γ = 0, 2, 8, 32) while the whole frame keeps moving
(mean|d| 2.0 -> 3.6 -> 4.4). Reason: a wall pixel's aligned tap has
`dot(n_p, n_q)` = 1 and its misaligned taps have 0, so `pow(., γ)` returns
the same value for every γ > 0 — there is no intermediate alignment for γ to
interpolate. The wall tile's actual knob is the plane kernel, not `BI_GAMMA`.
Measured against the old vertical-normal bake at γ = 32 (new-vs-old mean|d|
6.9, 50% of pixels past 0.05; scene mean luminance 67.7 vs 63.7, restoring
it toward the trilinear 68.5), so the baked normals were reaching the shade —
the wall's γ-invariance was not a wiring bug.

**The relight blend low-passes the SH field across space.** `blend_sh.wgsl` now
does the temporal cross-fade and, after it, a spatial pre-filter on the 9 SH
coefficients of each probe. The **default is a separable cubic B-spline
pre-filter** (1D mask `[1,4,1]/6`, tensor 3×3 `[1,4,1;4,16,4;1,4,1]/36`,
partition of unity 36); the previous **5-tap horizontal tent cross**
(`0.5*SH(x,z) + 0.125*[SH(x±1,z)+SH(x,z±1)]`) is kept behind
`const TENT_PREFILTER` in the shader for A/B. The point is the 1 m cascade-0
probe spacing, whose Nyquist ripple shows as columns/blobs; the stride-10 word 9
(`position, radius`) differs between probes and is copied through the plain mix,
not averaged, and neighbours are kept within the probe's own cascade and layer
and clamped at the grid edge, so floor/ceiling and cascade boundaries never
bleed. It is fused into the existing per-frame `blend_sh` dispatch (shade still
reads the same `sh_probes_blend_buffer`, so the shade pass is untouched) and
reads only the two raw relights, never its own output, so there is no
feedback/over-blur. The pre-filter body is kept commented at the bottom of the
shader. The on/off switch is a runtime toggle on `BlendToggleState`
(`src/ui.rs`; F key / `S_RC_LOWPASS`, default on = gated B-spline), extracted to
the render app where the blend dispatch writes it into `BlendUniforms`
(`dispatch_cascades` reads `Res<Res>…` — note the 16-system-param limit, which
is why the two march-cache `Local<bool>`s live in one `Local<Option<(bool,bool)>>`).
The **height gate is unconditional**: every neighbour tap is scaled by
`height_agreement_tol(q_y - c_y, spacing * LOWPASS_HEIGHT_TOL_SPACING)` (1.5)
and the surviving weights renormalise, so a height step sharpens the filter
back towards the center probe instead of dimming it — `height_agreement` itself
is 4 spacings (too loose to see the room's ~1.5 m platform step), hence the
tolerance-aware variant added alongside it in `heightmap_trace.wgsl`. The gate
was briefly a G-key toggle and is reverted: the current default is bit-identical
to the old `S_RC_LOWPASS_GATE=1` frame.

**Measured: the pre-filter (either kernel) does not move the vertical-seam
signature.** At the wall camera, `S_RC_LOWPASS=0` vs B-spline vs tent:
row-averaged vertical-line profile std 2.893 / 2.888 / 2.890 and seam energy
(per-column mean |Δlum|) 0.497 / 0.492 / 0.493 — indistinguishable. The B-spline
smooths the samples a little more than the tent (bspline-vs-nofilter mean|d|
0.324 vs tent's 0.241; bspline-vs-tent mean|d| 0.085, max 1.0, 22% past 0.05),
but a pre-filter only removes the grid's high-frequency ripple. The C⁰ gradient
kink at a cell boundary is a property of the **reconstruction**, so the other
half of the scheme — the cubic B-spline *basis* at shade time — is implemented
too (below).

**The shade can reconstruct with the cubic B-spline basis (`bsp_recon_enabled`,
C key / `S_RC_BSPLINE_RECON=1`, off by default).** `bspline_weights` in
`shade_cascades.wgsl` returns the four 1D B-spline basis weights
`B(t+1), B(t), B(1-t), B(2-t)` (C², sum to 1) at the raw in-cell fraction `ft`;
the branch reads a 4×4 XZ patch per probe layer (offsets -1..+2, clamped) instead
of the smootherstep-weighted 2×2, keeps the same two-layer vertical fade and
occlusion path, and accumulates coefficients mix-then-clamp like the default.
The default scene is bit-identical to the pre-reconstruction default (the diff
is only the widened UI panel, cols < 458); 0 shader errors.

**Measured: the reconstruction changes the frame but does not reduce seams
either.** At the wall camera, B-spline recon vs default: 58% of pixels change,
mean|d| 0.92, max 30.7. But vertical-line profile std 2.888 → 2.886, seam energy
0.492 → 0.498, and on the **isolated indirect term** (subtracting a direct-only
frame, `S_RC_INDIRECT=0`) 2.642 → 2.676; a flat-wall second-derivative roughness
rises (|∂²y| 0.66 → 0.84), and an FFT of a wall patch shows no probe-spacing
periodic peak (energy sits at ~0.34 cyc/px, i.e. texture detail, not the ~0.01
the ~1 m spacing would give). So neither interpolation order, pre-filter kernel,
nor reconstruction basis moves the seam signature — the seams are not produced
by the SH interpolation, and the reconstruction is left gated off.

**The toggle panel is behind `SHOW_TOGGLE_PANEL` (`src/ui.rs`); flip it false to
recompile the UI out.** The panel is drawn over the scene, so its button pixels
land in every screenshot and pollute an A/B diff — earlier low-pass numbers
(`05cc8126...` vs `135ebf38...`) included it. Scene-only, at the wall camera
(current default, ambient off, crop x > 460): filter-on vs `S_RC_LOWPASS=0`
mean|d| 1.29, mean luminance 53.87 vs 52.69 (md5 `38990d0f...` vs `27f374b9...`)
and `S_RC_LOWPASS=0` reproduces the pre-filter frame **bit-exactly** (scene
equals `135ebf38...`). The gate at 1.5 spacings is tight for this room: it drops
most neighbours, so the current default (gated filter) sits ~0.24 mean|d| off
the unfiltered frame (mean|d| 0.24, max 2.3, 1.0% past 1.0 vs `S_RC_LOWPASS=0`,
scene identical to the old `S_RC_LOWPASS_GATE=1` frame) — effectively the gate
switches the filter nearly back off, which is the knob to loosen if the step
bleed is the visible failing.

There is no mid-room probe layer any more; the old "probes mid-way between floor
and ceiling" state, the `S_RC_*` legacy leak camera and the Sponza numbers all
predate it.

**The probe display is instanced geometry now, not a fragment-shader trace.**
The old `probe_spheres` (a 2D DDA per cascade walking the probe grid along each
view ray in `shade_cascades.wgsl`) is deleted, along with the `probes_enabled`
field of `ShadeUniforms`; `shade` is indirect-only and no longer stays alive for
the P toggle. In its place `src/probe_overlay.rs` + `assets/shaders/
probe_overlay.wgsl` rasterize a unit icosphere (subdivision 2, 320 tris) once per
probe-with-layer - one `draw_indexed` instance each, 3678 instances total - in a
custom **opaque** pass that draws into the **window camera's own framebuffer**
and tests against **that view's prepass depth** (reverse-Z, `GreaterEqual`,
depth **write on**), so geometry occludes the spheres *and the spheres occlude
one another*. Depth write was off at first: every sphere fragment then passed
the scene-depth test and overlapping spheres overwrote each other in instance
order, which read as "weird blending" - the four cascades stack on the same
surface, so there is a lot of overlap. The per-instance world
position is `probe_positions[2*i].xyz`, the same buffer trace/gather/shade read,
so the spheres sit exactly on the baked two layers (floor + ceiling at their real
surface heights, `escape_texel` nudges included) with nothing re-derived on the
CPU. The buffer is already concatenated cascade-major in `sh_base` units, so an
instance's cascade - and its radius, `spacing * PROBE_RADIUS_SCALE` (0.12,
Rust-side) - is found by walking the per-cascade `(first, count, radius)` entries
in the uniform, packed from the CPU-side `CascadesLayout`. Each sphere is
**shaded by the SH probe it stands for**: instance `i` reads its 9 coefficients
at word `i * SH_STRIDE` of `sh_probes_blend_buffer` (the same cross-faded buffer
the shade pass samples) and evaluates the clamped-cosine `probe_irradiance`
along the sphere's own world normal (`= local_pos`, since the unit icosphere is
placed with a positive uniform scale and no rotation). `PROBE_BRIGHTNESS` (1.0,
in the shader) scales the pre-tonemap write. The old per-cascade palette
(green/blue/purple/yellow) and additive blend went in and came back out: with a
flat per-cascade tint and `BlendFactor::One/One`, a cell's two coincident-or-
stacked layers of the same cascade composited to the *same* bright colour, so
the grid read as a **single layer**; the per-probe SH shading makes the layers
distinct (floor-layer probes light toward the ceiling and usually read
blue/sky-lit, ceiling-layer probes read dark under the ceiling). The pass is
**gated three ways**: `P` /
`S_RC_PROBES=1` (`LightingState.probes_enabled`), `BakeState::is_done()` (nothing
to draw until `probe_positions` is written), and the window target
(`ExtractedCamera.target` must be `NormalizedRenderTarget::Window`) - the two
startup bake cameras are live 512x512 `RenderTarget::Image` views of their own,
and without that last guard the overlay also splatted spheres into their targets.
`draw_probes_gizmos` in `main.rs` is still dead behind its original `return;`
(it sat spheres at a fixed y = 1.5, never the real layers) and is superseded by
this pass; kept only because the workstyle forbids deleting machinery to turn it
off. The world->clip matrix is `clip_from_view * world_from_view.inverse()`:
`ExtractedView.world_from_view` is the camera's **world** matrix (`world <-
view`), so world->view needs the inverse - using `world_from_view` directly (as
an earlier cut did) made the spheres track an arbitrary camera-derived transform
that read as them "following the camera". The icosphere is drawn with **all** its
indices (`indices.len()`); an earlier cut passed `indices.len() / 4`, which drew
only the icosahedron's first **5 of 20 faces** - one contiguous pentagonal cap -
and that is what rendered as the "weird hemispheres", not the probes being
buried. Measured at the interior camera
(`S_RC_CAM=3.50,0.73,3.47,1.84,-1.13,-6.22`, sun pinned 45,54.7, ambient off):
with the cap and radius scale 0.3, probes on vs off changed 31.75% of pixels;
fixing the index count alone (full spheres, 0.3, depth write still off) jumped to
**79.5%** - the four cascades stack on the same surface and every fragment
overwrote in instance order, so the grid became a solid mass. With depth write on
and radius scale 0.12 it is **21.2%** and the spheres read individually. The
overlay luminance spans ~40 to ~240, so the SH read reaches the spheres. 0
shader errors; the indirect path is otherwise untouched (with probes off the
removed branch was already unreachable, and the surviving `depth <= 0 ||
!indirect` guard is unchanged), so a probes-off frame is identical to the
pre-change one.

**The probe nudge is a minimal push out of the solid, not a relocation.**
`probe_position_baked` and `probe_normal_baked` test a sphere of the cascade's
ray-interval **start** (0/2/4/8 m) against the winding solid
(`nudge_out_of_solid`) and push the probe off the nearest overlapping texel by
exactly the overlap (`r - dist + PROBE_SURFACE_EPS`), relaxed over 3 passes;
`probe_normal_baked` takes the push direction as the normal. Cascade 0's
interval starts at 0, so it never nudges. This replaced the old
`sphere_is_buried(xz, r)` + `escape_texel_xz(origin)`: the escape returns the
origin column when it is not solid, so a probe whose gather sphere clipped a
neighbouring wall never moved - the burial test and the escape disagreed - and
it relocated a buried probe a whole cell to the nearest open column instead of
moving it the centimetres an overlap needs. `escape_texel_xz` survives as the
inside-solid hop; the old body is kept commented at the call site. Measured at
the interior camera, probes off, new vs old (`r = PROBE_LAYER_GAP`): **74.3%**
of pixels change, mean |d| 8.43, scene mean luminance 52.55 -> 50.61, and the
user accepts the result. The display radius is back to `spacing *
PROBE_RADIUS_SCALE` (0.12): sizing it by the interval's end (2/4/8/16 m) drew
spheres that occluded the room (82.7% of pixels, mean luminance 50.6 -> 128).
0 shader errors, bake verified (`79938 ceiling hits`).

The earlier winding-only `is_solid` swap, the `Heightmap::solid` /
`bake_caps.rs` deletion and the `EMPTY_FLOOR` sentinel fix (the `usable` test's
`h.y > uniforms.split - 9.0`) are committed. `probe_visible` is an occlusion
trace again.

## March-once, relight every 4, and the blend pass that made it not cost anything

Three things are in now: the ray march runs **once** into `hits_buffer`
(`march_hits.wgsl`, and `cascades2_5d.wgsl` reads hits instead of tracing);
light/trace/gather relight every `RELIGHT_INTERVAL = 4` frames; and the sun
sweeps on a 20 s period (`animate_sun`, `S_RC_SUN_STILL` freezes it). Three
rotating SH probe buffers cross-fade the gap: each relight gathers into the
phase's `G`, and the phase advance makes `G`→`A`, `A`→`B`, `B`→`G` via the
`A_OF_PHASE`/`B_OF_PHASE`/`G_OF_PHASE` tables, so no relight is ever copied -
the fresh one enters the blend at weight 0, climbs to 0.25/0.5/0.75, and hits
1 "by itself" on the next sync when the swap turns it into `prev`.

**`S_RC_INDIRECT`, `S_RC_PROBES` and `S_RC_OVERLAY` used to do nothing unless
`S_RC_SCREENSHOT` was also set.** They are read on frame 2 unconditionally now.
Every timing taken before that fix was "indirect on, debug planes rendered" no
matter what the command line said, which is why `indirect=0` and `indirect=1`
came out identical. `S_RC_SUN_STILL` freezes for *any* value including empty
(`is_ok()`), so unset it with `env -u` when you want motion.

### Timing needs a HEAD binary running HEAD's shaders

Shaders load from disk, so a stashed/HEAD binary pointed at the working tree's
`assets/shaders/` fails validation and exits **without printing one `FPS` line**
— only `CommandQueue has un-applied commands` warnings. Build a second
`BEVY_ASSET_ROOT` root of symlinks with the changed `*.wgsl` restored from
`HEAD`, and give both binaries the env fix, or the A/B is not the same scene.
(The old build also prints `46056 cap texels`, which HEAD's caps still have.)

### Doing the cross-fade in the fragment shader cost 45 ms

`shade` samples 4 probes x 32 hemisphere directions x 9 SH coefficients, so a
per-coefficient `mix(prev, curr, alpha)` doubled the biggest storage read in the
app, 1.44 million pixels at a time:

| | frame | no shade | shade |
|---|---|---|---|
| HEAD | 70.9 ms | 33.7 ms | 37.2 ms |
| march + relight, mix **in shade** | **116.3 ms** | 32.1 ms | **84.2 ms** |
| ...after `blend_sh.wgsl` | 76.3 ms | 32.1 ms | 44.3 ms |

`assets/shaders/blend_sh.wgsl` mixes the two relights over the whole probe set
(1839 probes x 10 words) once per frame, immediately after the gather, and
`shade` reads that single buffer. It is the only work on a non-relight frame
besides the alpha uniform write. `Relight::alpha` went with it: `shade` no
longer has a `relight_alpha` field or a fifth binding.

Second, `probe_visible` and the 32-sample integral now sit behind `if w > 0.0`,
skipping both for a probe that is out of footprint, out of height agreement or
buried. It is provably free (`w * finite == 0`) and measured **bit-identical**
— static-sun md5 `2c9e0449241b9a693b3072c5ebc25a11` unchanged — but it moved
nothing on this camera, so the probes are apparently not often at zero.

### Where it stands

Full frame is at parity with HEAD (76.3 ms both), the shade pass is 4 ms
*cheaper* than HEAD's, and the non-relight path is pinned at 32.1 ms across five
runs where HEAD's no-shade runs oscillate 27.5 / 34.0 ms — enough that HEAD's
"shade" column is only good to about ±6 ms. Shade is still 58% of the frame, so
the next real win is inside it (`probe_visible`'s trace, or `probe_buried`,
which HEAD does not have at all), not around it.

Five interleaved runs of 35 s each, median of per-run medians, first three `FPS`
samples dropped, `S_RC_OVERLAY=0 S_RC_SUN_STILL=1 S_RC_INDIRECT` varied, leak
camera `-10.58,4.86,-3.21,-1.17,8.18,-3.83`.
