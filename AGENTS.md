# Working notes

Handoff for whoever picks this up next. Delete freely — the git history carries
the reasoning. Kept short on purpose; ARCHITECTURE.md and EXPLAINED.md were
dropped before because they outlived the code they described.

## What this is

Bevy 0.19 indirect lighting over a dual-channel heightmap (floor + ceiling per
column, `SPLIT = 5.0`). Sponza, baked once at startup into textures the cascade
probes then trace against.

## Commands

```sh
cargo build --bin s_rc
cargo clippy --bin s_rc

# Leak camera (the one that matters for the split):
BEVY_ASSET_ROOT=$PWD S_RC_CAM="-10.58,4.86,-3.21,-1.17,8.18,-3.83" \
  S_RC_INDIRECT=1 S_RC_OVERLAY=0 ./target/debug/s_rc
# Second storey:
#   S_RC_CAM="-9.20,4.88,-6.88,-0.63,6.62,-4.79"
# S_RC_PROBES=1          show probes
# S_RC_OVERLAY=1         keep the heightmap debug planes
# S_RC_SCREENSHOT=/tmp/x.png    also exits at frame 240
# Dump any baked image whole, to poke at it in numpy:
#   S_RC_DUMP_HEIGHT=/tmp/h.bin     Rg32Float  512x512x2, **x = ceiling, y = floor**
#   S_RC_DUMP_META_MIN=/tmp/m.bin   Rgba32Float, xyz = ceiling normal, negated
#   S_RC_DUMP_META_MAX=/tmp/m.bin   Rgba32Float, xyz = floor normal, not negated
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
  `S_RC_PROBES=1` also changes pixels, so keep it off for A/B shots.

## State

Uncommitted working tree. The inside-wall test is now **derived from the two
stored normals** instead of a dedicated channel, and the cap machinery is gone.

`heightmap_trace.wgsl`'s `is_solid` flags a column when either stored normal has
`y < -0.5`, checked over the probe's whole 2x2 bilinear stencil
(`stencil_touches_solid`). The bake cameras have material culling off, so they
rasterise backfaces: a column in open air records the surface its ray *lands* on
(floor up, ceiling down), while a column that starts inside material records the
surface it *leaves* through — the underside going down, the top going up, both
facing away from the split. The bake negates the ceiling normal so both channels
land on the same sign. `Heightmap::solid`, its bind-group slot and its bind-group
entry are deleted; `src/bake_caps.rs` is deleted with them.

Measured on Sponza, no caps:

| | |
|---|---|
| ceiling hits / open sky | 102226 / 159918 (61% of texels) |
| ceiling heights | 6.5–7.0 m and 10.5–11.3 m |
| floor heights | 0 m (ground), 2.5–4.5 m, 4.6–4.9 m |
| flagged columns | 2300 texels = 11 m² = 0.9% of the map |

### The test was indexing the wrong lattice, and the validation below was measured through that bug

`SPACING0 = 1.0` m per probe, but the heightmap is 512² over 40×32 m, so a probe
grid index and a heightmap texel differ by **12.8x in x and 16x in z**. The shading
pass was passing `cell + corner` (probe grid) straight into
`stencil_touches_solid`, which `textureLoad`s the heightmap. Cascade 0's grid only
spans indices 0..40 / 0..32, so the mis-indexed lookup could reach 1353 texels —
world x −20..−16.9, z −16..−14.0 — and **0 of the 2300 flagged texels are in
there**. The test was a no-op at every pixel in the scene. Fixed by
`heightmap_trace.wgsl`'s `probe_in_solid(c, probe)`, which makes the probe-grid →
texel round trip through the same world-space step `probe_position` uses, and called
with the clamped `q` the probe is actually read from. Effect at the leak camera is
199 px of ≤1/255, all in the left third.

So the two measurements below are evidence about the *threshold*, not about the test
being wired up: forcing `BURIED_MIN_Y = 1.5` moved the leak shot by 88.25% of pixels
because flagging *everything* flags the reachable corner too. At the real threshold
the leak and storey shots are bit identical with the test on and off, because there
are no wall columns at the split to find. Two independent checks agree:

- Off the glTF: the cross-section of the whole mesh at y=5.0 is 730 segments and
  they all come from one small object; the even-odd fill of that cross-section
  encloses **0 texels**. Parity is meaningless on this asset anyway (Sponza's
  shells are open — the fill reports 0 enclosed at every height, including ones
  that are obviously inside the building), so do not trust it as an inside test.
- Per-column vertical ray crossings: sampled columns through the perimeter walls
  cross two surfaces, both *below* the split (0.05 and 4.65 at x=-14,z=0; 0.2 and
  4.02 at x=-13.5,z=0). The slab above spans 4.62–5.38 where it exists.

So `SPLIT = 5.0` sits in the 0.7 m band at the top of the ground storey, between
the wall tops (~4.65) and the next storey's structure (~5.38), and mostly inside
the slab that band contains. The 2300 flagged texels are exactly those slab
columns: the 1160 that only the ceiling channel flags have floor 4.06 / ceiling
5.56 (p50) — a 1.5 m interval straddling the split whose stored top face points
up, which is what a column starting inside the slab must read. The test is
therefore validated on real buried columns; there are just almost none.

### The test is sensitive to the split's exact height

The bake cameras sit *on* the split with `BAKE_NEAR = 0.05`, so a surface whose
face is within 5 cm of the plane is clipped away and the ray continues to the far
side of whatever it was part of — reading as "started inside material". Baked at
`SPLIT = 8.0`, whose upper-storey slab tops sit at ~7.95, the flag explodes from
0.9% to **10.1%** of the map (26478 texels) and the flagged columns no longer look
buried: their stored interval is ~4.0 m (floor 6.85, ceiling 10.82), i.e. slab top
just below the plane and roof above it. So a split chosen to graze a big slab
flags that slab's whole footprint. It also means the test cannot distinguish
"inside material" from "nearest surface on that side is steeper than 60° from
horizontal" — stair soffits, vaults and sloped ceilings all trip it. If a split
is ever chosen again, keep it clear of nearby surfaces by more than the near
clip, and expect steep geometry to flag.

### The caps were the wall-probe leak

Worth recording, because it inverts the reason the problem was thought to need a
stencil pass. The caps put a synthetic ceiling at 5.052 over the perimeter band,
whose columns' floors are the storey-2 floor at ~3.5 — so the stored interval ran
3.5 → 5.05, straight through the perimeter walls, and the probes were lifted into
the masonry. Without caps those columns keep their real ceiling (6.5–11.3 m) and
their probes are lifted off the wall tops instead, which is above the material.

Also correcting the previous note: a capped bake does **not** fill 111200 ceiling
texels. Capped and uncapped both fill 102226 — the caps replaced 46056 *existing*
ceilings 5 cm lower rather than closing any leak, which is why "the leak is
closed" never showed up in the numbers.

## Next step

**The open-sky columns, not the walls.** Counting live probes per camera from
the dumps (probe centre on the cascade-n grid, `spacing(0) = 1 m`, radius from
`RADIUS0`), 44–54% of the live probes around both cameras sit in columns whose
ceiling is `NO_CEILING`: 149/340 and 181/338 in cascade 0, rising to 70% by
cascade 3. Those probes are clamped to a short lift above a floor sentinel and
have nothing above them, so the storey over the atrium and the exterior are
absent from the map entirely. At the same time the storey slab sits *inside*
other columns' intervals, invisible to the heightmap. Both are the single
interval limitation below, and neither is a probe-placement bug — which is why
the normal test, the caps and a stencil pass all leave the image untouched.

Pick one, it is a design call rather than a bug:

- **Split placement.** Anything that puts the plane inside a storey's air gives
  the wall test something to reject, but costs the map the other storey, and the
  5 cm near clip makes grazing heights pathological (above).
- **Multi-frustum.** Up to three artist-controlled up/down intervals per column
  is the design that actually holds two storeys. Bigger, and it is the one the
  notes have been heading towards.

### The over-bright ceiling at `S_RC_CAM=-5.89,5.41,2.86,3.55,6.50,5.95` is a sky leak, not a solid test

That camera looks down a 3 m room whose roof is the underside of the upper slab at
6.88 m (columns floor 3.32 / ceiling 6.88). The ceiling takes the strongest indirect
in frame, +20..+23.5 of ~48 total luminance, and it is *not* rejected by the normal
test: the four columns it is sampled through are unflagged (both normals +0.9..+1.0).

The cause is that the probes the ceiling samples — lifted to 3.32 + min(0.5·3.56,
2.0) = 5.10 m, 1.78 m below the pixel, inside `HEIGHT_TOLERANCE_SPACING` 2.0 m —
are lit as if they were outdoors. Traced the way `gather_cascades` marches:

| probe at (−3.0, 5.10, 3.0) | reaches open sky |
|---|---|
| heightmap | 62% of its upward hemisphere (40/64 cosine rays) |
| the glTF itself, same rays | **0%** — 32/32 rays hit geometry at 1.35–5.26 m |

Across all 36 live cascade-0 probes within 8 m of the camera it is 23% vs 5%. Every
escaping ray leaves through the same place: world z ≈ 8.83, where the model ends. A
vertical wall is invisible to a floor/ceiling map — the bake's rays run straight up
and down and never cross one, so both sides record open air and no column is
flagged. That is the same missing-information problem as the two storeys above, and
it is why the ceiling is bright rather than the wall leaking.

**Relocation is implemented, and it is blind to exactly this.** `probe_position`
now resolves its column through `escape_texel` before lifting: if `is_solid`
fires, it searches every ring up to a per-cascade reach for the open column
that puts the probe *closest* to where it is now. `probe_buried` (the shading
pass's test) asks about the probe's actual position rather than its grid cell,
since testing the cell would reject precisely the probes the nudge rescued.

The rings grow in Chebyshev steps, but that is only the enumeration: every
candidate in every ring is scored by the full 3D move in **metres** — texels
are 7.8 cm by 6.2 cm, so texel-space distance over-weights x by a quarter and
ignores the height axis a buried probe most often has to move along — and the
smallest wins. Stopping at the first ring with anything open would answer
"nearest by square" and settle on a texel up to sqrt(2) further off.

Reach is `max(0.75 m, min(interval.x, spacing/2))` = 0.75, 1, 2, 4 m for
cascades 0–3, from each cascade's own `interval.x` (0, 1, 2, 4 m): a coarse
cascade's rays start beyond a wall cascade-0's still run into, so it can afford
to send the probe further, and `spacing/2` caps it because past that the probe
has left the bilinear footprint that reads it. `column_open` refuses a
destination with no recorded ceiling — relocating a buried probe out into an
open-sky column would pour outside light into the room it was rescued from.

Verified live by mirroring the whole thing off the dumps
(`/tmp/opencode/solid/reloc_check.py`): **2 cascade-0 probes of 1353** sit on a
flagged column, both escape, neither is stuck, and both move sideways —
0.113 m and 0.156 m, 100% horizontal, vertical 0.016 m median. Cascades 1–3
have no probes on flagged columns at all. So the machinery is correct and does
something, and it is still nowhere near enough: `is_solid` flags 0.9% of the
map and the probes that need rescuing are not on it.

Each bake camera already has its own G-buffer — `bake_target` adds a distinct
image handle per camera and `ImageRenderTarget` equality compares handles, so
the up and down views cannot share attachments and clear over each other. That
is why `is_solid` can read both normal buffers, and it is not the limitation.

### Two measured leaks, one cause: thin mat-5 panels the bake cannot see

Both confirmed cameras have the same signature, and neither is fixed:

| camera | leaking surface | dist | column reads | is_solid | adds |
|---|---|---|---|---|---|
| `-7.02,3.62,-6.46,2.51,2.03,-9.05` | mat 5, z=−9.00, n=+z | 2.5 m | −0.92 → 10.73 | n | +39.8 mean, +66.8 max, 40756 px |
| `-9.20,4.88,-6.88,-0.63,6.62,-4.79` | mat 5, z=−6.26, n=−z | 1.3 m | −0.92 → 11.20 | n | +63.2 max, 26380 px |

Both panels are ~0.1 m thick and sit in a column whose stored interval is 12 m
of apparently open air, with both stored normals pointing up — the vertical bake
rays pass beside them and exit through the roof, so the column reads as open sky
and nothing flags it. Probes lifted into that column then gather sky from the
open-sky columns on the far side of the panel. Measured on the z=−9.00 probe at
(−1.5, 1.08, −9.0): 32/64 upward rays reach "sky" in the heightmap, 0/64 in the
glTF, whose median nearest hit is 0.08 m against the heightmap's 10.03 m.

Neither camera is inside material: 200 Fibonacci directions give a nearest hit of
0.202 m (z=−9 run) and 0.622 m (storey-2 run), with 0/200 and 0/200 inside 0.3 m.
The storey-2 camera looks along a wall 0.62 m away, which is what fills its frame.

This is the same missing information as the two storeys above and the open-sky
columns: a floor/ceiling map has no way to hold a vertical surface. Fixing it
needs a horizontal signal — extra bake views, a multi-frustum interval, or a
depth-based probe relocation like DDGI's — none of which is per-probe nudging.

### 115 probes are visibly inside the wall, and none of them can be flagged

Measured by testing **every** stored probe position against the glTF
(`/tmp/opencode/solid/all_inside.py`, no camera maths, 8-way nearest-geometry
test on all 1839 probes across the four cascades). "Inside" means nearer than
the sphere the probe is *drawn* as, `0.15 * spacing`, so these are the ones
visible in `S_RC_PROBES=1`:

| cascade | probes | sphere r | inside sphere | within 0.35 m | `is_solid` flags |
|---|---|---|---|---|---|
| 0 | 1353 | 0.15 m | 70 | 134 | **0** |
| 1 | 357 | 0.30 m | 32 | 33 | **0** |
| 2 | 99 | 0.60 m | 10 | 8 | **0** |
| 3 | 30 | 1.20 m | 3 | 0 | **0** |
| | 1839 | | **115** | 175 | **0** |

Zero. Every one of the 115 is in a column `is_solid` calls open air, so
`escape_texel` never fires on any of them and relocation cannot help — not
because the search is wrong, but because there is nothing to search from.

Cascade 1 is the cleanest proof. Fourteen probes at `(x, 1.08, −9.00)` for
x = −13…+13 sit **0.001 m** from a mat-5 surface — millimetre-exact *on* the
z=−9.00 wall plane, which is the leaking wall of the first camera above. Their
columns read −0.92 → 10.73, unflagged. Cascade 2 has the same six probes along
z=−6.00 with nearest 0.300 m, hitting the z=−6.26 wall of the second camera.
Cascade 0 has nine in a row at x=−11.50, also mat 5.

### The cause is `PROBE_MAX_LIFT`, not the missing wall

1187 of the 1353 cascade-0 probes are clamped at the top of the lift range, and
they land on only **three** distinct heights:

| y | count | derivation |
|---|---|---|
| **1.08** | 282 | ground −0.92 + `PROBE_MAX_LIFT` 2.0, in columns spanning ~12 m (ceil 10.8–11.3) |
| **1.98** | 45 | ground −0.02 + 2.0, `NO_CEILING` |
| **−3.00** | 860 | `EMPTY_FLOOR` −5.0 + 2.0 |

The first is the leak. A column that reads 12 m of open air gets a lift of
`0.5 * span`, which clamps to `PROBE_MAX_LIFT`, so the probe is placed 2 m above
the ground — and Sponza's perimeter walls span y≈0.05…4.65, so **1.08 m is dead
centre of storey 1**. The clamp is not failing to avoid the wall; it is
actively aiming the probe at it, for all 282 of them at once.

The third is worse and is a plain bug rather than a design tradeoff.
`EMPTY_FLOOR` −5.0 is a *sentinel* for "the down-facing ray found no floor at
all" — the columns off the end of the model, where the bake legitimately records
nothing. `lifted_height` does arithmetic on it as if it were a surface, so 860
probes (64% of cascade 0) are placed at −3.0 m, three metres *underneath* the
model. Nothing is ever below there, so it is not visibly wrong, but the height
is an artefact of two sentinels added together and means nothing. Clamping a
*sentinel* upward is never defensible; that one can be fixed without a horizontal
signal.

A second, smaller flaw: probe `(16,17)` at cascade 0 **does** relocate (it is in a
flagged column) and still ends up 0.043 m from a mat-12 surface, because
`column_open` only knows the heightmap and cannot tell that the destination
column's own lift is flush with a wall.

### Backface culling during the bake destroys the containment signal — don't retry

Worth recording because it is the obvious thing to try and it is exactly
backwards. `BAKE_CULL_MODE` in `src/heightmap_bake.rs` was set to
`Some(Face::Back)` (the bake previously ran with culling off, see that
function's doc comment) to see whether not rendering backfaces would stop a
camera sitting inside a slab from recording its own exit face.

It removes the signal completely: **flagged texels go 2300 → 6**, 0.88% →
0.00%. The 2294 that stop being flagged are the ones that mattered, and their
intervals before culling were floor p50 −1.01 / ceiling p50 10.70 — the tall
open-air columns the walls hide in.

The reason is the same geometry that hides the walls. With culling off, a camera
inside a slab looking up hits the slab's top face *from behind*: that face's
normal points away, which is the whole detection mechanism. With back faces
culled, that exit face is culled, the ray continues past it, and records
whatever is beyond — a normal pointing back down at the camera, i.e. exactly the
open-air signature. Culling does not stop the bake seeing through material; it
stops the bake recording that it is inside it. Containment detection and
backface rasterisation are the same mechanism, not independent choices.

Other effects, all consistent with that: floors recorded as `EMPTY_FLOOR` rise
159839 → 161219 texels (more of the model becomes invisible), 1273 ceilings and
1512 floors change value, and the leak camera is essentially unchanged
(mean |Δ| 0.014 over 1600×900, luminance 33.70 both ways) because the wall
columns it leaks through were never flagged in the first place. **Culling is
still the right default for a bake** — a single-sided surface whose normals face
away would otherwise be skipped and the floor/ceiling recorded from some deeper
surface. Only the containment test wants the opposite, and it gets that from
backfaces.

### Front-face culling is worse — the containment test needs *both* directions off

`BAKE_CULL_MODE = Some(Face::Front)`, the third and last cull mode tried:

| | none (current) | `Face::Back` | `Face::Front` |
|---|---|---|---|
| flagged texels | 2300 (0.88%) | 6 (0.00%) | **110656 (42.21%)** |
| cascade-0 probes inside 0.15 m | 70 | — | **124** |
| probes within 0.35 m | 134 | — | **182** |
| probes relocated | 2 | ~0 | 109 |
| relocated *but still inside* | 1 | — | 54 |
| leak camera luminance | 33.70 | 33.70 | 28.30 |
| mean abs delta on leak camera | — | 0.014 | 5.399 |

It does flag all three wall columns, which is the one thing it gets right — but it
flags 42% of the map, and 43764 of those (40%) are open-sky or `EMPTY_FLOOR`
columns flagged purely because *nothing was recorded*: `EMPTY_FLOOR` rises
159839 → 186139 and 100179 ceilings and 89914 floors change value. `is_solid`
stops meaning "inside material" and starts meaning "the bake saw nothing", which
inverts the signal instead of sharpening it.

The mechanism mirrors the backface case. With front faces culled, a camera in
open air looking up skips the first surface it meets and records the first back
face beyond it — a normal pointing away, which reads as buried. So **open air
reads as buried**. Both extremes fail in the same way, for opposite reasons:
`Face::Back` records nothing (2300 → 6), `Face::Front` records the wrong thing
(2300 → 110656). The test only works with culling off, where a camera inside
material records the exit face behind it and a camera in open air does not.

Relocation also gets worse: it now fires on 109 probes instead of 2 and 54 of
them land inside geometry anyway, because `column_open` can only judge a
destination from the same heightmap that is now wrong almost everywhere.

All three modes are now measured. Reverting is `BAKE_CULL_MODE = None`.

### Tuning the lift does not fix it — do not try it again

`/tmp/opencode/solid/lift_experiment.py` re-runs the whole field with
`PROBE_MAX_LIFT` as a parameter:

| lift | inside 0.15 m | within 0.35 m |
|---|---|---|
| 2.0 (current) | 70 | 134 |
| 3.0 | 70 | 134 |
| 4.0 | 70 | 133 |
| none (half span) | 71 | 133 |

No value helps. The nine probes at x=−11.50 go from y=1.08, inside a perimeter
wall, to y=5.20, inside the storey-2 slab — nearest geometry 0.072 → 0.050 m.
There is a surface at nearly every height in that column and the map only
records two of them, so any choice of height is a coin flip. This is the
missing-information problem above, not a tuning problem: the 12 m column has
walls in it that no floor/ceiling bake can have recorded.

Fixing it needs a horizontal signal in the bake — extra views, a multi-frustum
interval, or a depth-based relocation like DDGI's. The user has ruled out extra
cameras, so the remaining options are the latter two.

### 2D generalized winding number on the split plane — implemented, and it beats the normal test

The horizontal signal the bake has been missing, taken straight from the
geometry with no extra cameras. `src/winding.rs` cuts every scene triangle at
`SPLIT`, orients each segment by `cross(+Y, n)`, and sums
`atan2(cross, dot)` over a 512² grid of texel centres:

    w(p) = (1/2pi) sum atan2( cross(a-p, b-p), dot(a-p, b-p) )

`is_solid` gained a third term: `abs(textureLoad(winding, t, 0).x) > 0.5`.
The map is group-0 binding 4, so every cascade pass and the shade pass see it
through the existing `common_bind_group`.

- **The Rust field is the reference, bit for bit.** Against
  `/tmp/opencode/solid/gwn2d.py` at the same texel centres: max |diff|
  7.6e-6 (f32 rounding), 100.000% agreement on the `|w| > 0.5` decision over
  all 262144 texels. Dump it with `S_RC_DUMP_WINDING=/tmp/w.bin`.
- **Use the atan2 sum, not a scanline crossing count.** The scanline version
  is O(rows + segments) instead of O(texels x segments) and is *wrong here*:
  Sponza's shells are open, so the sum is fractional, and the two disagree on
  5% of the plane — the crossing count loses 167 of the 225 points the atan2
  form flags. Likewise do not coarsen the grid: bilinear from 64², 128² or
  256² drops the catch of the normals' own 2300 texels from 98.5% to ~45%.
- **It dominates the normal test on the 115 probes inside their drawn
  sphere** (`/tmp/opencode/solid/wn_eval.py`, 1839 probes, ground truth is
  nearest geometry < 0.15·spacing):

  | test | catches of 115 inside | false positives of 1400 open |
  |---|---|---|
  | `is_solid` normals only | **0** (0.0%) | 0 |
  | winding at `SPLIT` = 5.0 (as shipped) | **32** (27.8%) | 0 |
  | winding at each probe's own height | **45** (39.1%) | 0 |

  Flagged texels go 2300 → 14222 (5.4% of the map), and the winding term
  reproduces 98.5% of what the normals already had — 11922 are new. Probes
  on a flagged column go 2 → 67 (44/17/6/0 by cascade).
- **The split-plane cut still cannot see a wall that tops out below it.**
  Sponza's perimeter walls span y≈0.05…4.65, so `SPLIT = 5.0` is above them
  and the cut only catches the slabs and pillars that actually cross the
  plane. This is why the two measured wall-leak cameras
  (`-7.02,3.62,-6.46,2.51,2.03,-9.05` and `-9.20,4.88,-6.88,-0.63,6.62,-4.79`)
  come out **unchanged**: A/B against the same build with `WINDING_SOLID`
  forced to 1000, they move 0.05% of pixels, max |Δ| 0.012 — dither level.
  The split camera `-10.58,4.86,-3.21,-1.17,8.18,-3.83` does move: 21.9% of
  pixels, mean |Δ| 0.0024, max +0.048, 6.7% brighter / 0.18% darker.
  A cut at each probe's own height sees those walls (4395 segments at y=1.08
  against 730 at y=5.0) and is what would take the catch from 32 to 45; it
  needs a stack of slices rather than one, which is the obvious next step.
- 730 segments is the whole cross-section at y=5.0 — one small object, as
  noted above. The field is sharp: only 14187 texels exceed 0.5, in bands at
  rows 110, 156-173, 217-223, 279-285, 330-341, 387-388.

## March-once, relight every 4, and the blend pass that made it not cost anything

Three things are in now: the ray march runs **once** into `hits_buffer`
(`march_hits.wgsl`, and `cascades2_5d.wgsl` reads hits instead of tracing);
light/trace/gather relight every `RELIGHT_INTERVAL = 4` frames; and the sun
sweeps on a 20 s period (`animate_sun`, `S_RC_SUN_STILL` freezes it). A double
buffer of SH probes cross-fades the gap.

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
