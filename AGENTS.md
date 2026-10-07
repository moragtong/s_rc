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

## Commands

```sh
cargo build --bin s_rc
cargo clippy --bin s_rc

# Legacy leak camera from the Sponza era:
BEVY_ASSET_ROOT=$PWD S_RC_CAM="-10.58,4.86,-3.21,-1.17,8.18,-3.83" \
  S_RC_INDIRECT=1 S_RC_OVERLAY=0 ./target/debug/s_rc
# Second storey:
#   S_RC_CAM="-9.20,4.88,-6.88,-0.63,6.62,-4.79"
# S_RC_PROBES=1          show probes
# S_RC_OVERLAY=1         keep the heightmap debug planes
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
  `S_RC_PROBES=1` also changes pixels, so keep it off for A/B shots.

## State

Probes sit exactly mid-way between their column's stored floor and ceiling
(`probe_position` returns `0.5 * (ceiling + floor)`, using the column it was
nudged to; only open-sky/empty columns fall back to the bake plane). The escape
nudge gates on `is_solid`, winding-only. `probe_buried` rejects a probe's
weight directly; `probe_visible` goes through `trace`, which consults the same
`is_solid`. `SPLIT = 0.0` is exact mid-room for `room.glb` (its 20×6×20 m
inner block runs y −3.0..3.0, interior −2.7..2.7).

Uncommitted working tree. The inside-wall test is now **derived from the
winding slice at `SPLIT`** and routed solely through `is_solid` (winding-only).
The `column_buried`/`column_open`/`lifted_height`/`escape_texel` normal-buffer
machinery stays in `heightmap_trace.wgsl`, unreferenced. `Heightmap::solid`,
its bind-group slot and its bind-group entry are deleted; `src/bake_caps.rs`
is deleted with them. `probe_visible` is an occlusion trace again.

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
