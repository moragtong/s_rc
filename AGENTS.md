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
# S_RC_DUMP_HEIGHT=/tmp/x.bin    dump the baked map (Rg32Float, 512x512x2)
# S_RC_SCREENSHOT=/tmp/x.png
```

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

## State

Two commits, both verified:

- `db4bf8e` — **authoritative verifier**. The bake used to be judged by size,
  which passed a stale uncapped map (102226 ceiling texels) as readily as a good
  one (111200). It now demands cap texels at the cap plane. 13/13 runs verify
  identically on both cameras.
- `2963a64` — **`Heightmap::solid`** channel and march-time probe rejection.
  Inert: written as all-zero, so nothing changes yet. Proven live by forcing it
  to 1.0 (89.6% of pixels change, mean luminance 32.6 → 27.6).

### Dead end, do not build on it

`src/bake_caps.rs` closes **no leak** — ceiling hits sit at 102226, exactly the
uncapped baseline. Its module header explains why. The machinery is sound and
reusable; the *capping* is what fails. It decides what is solid from surface
winding, which cannot work here: a vertical ray through a hollow wall and a
vertical ray through a tall atrium produce the same floor, ceiling and normals.
The heightmap has no way to represent "this column is inside material", which is
also why probes visibly sit inside wall hollows.

## Next step

**Test whether `solid` needs a stencil pass at all.**

The bake cameras have material culling disabled, so they rasterise backfaces.
For a column inside a wall, the down camera's ray travels down through the wall
and exits through its **underside** — so the first surface it records has a
normal pointing **down**. A column in open air lands on a floor pointing **up**.
Symmetrically for the up camera and the ceiling normal.

If that holds, `solid` is derivable from the signs of two normals the reduce
pass already writes, and the stencil pass is unnecessary. That is a shadow-volume
inside test obtained for free.

Experiment: spawn the `meta_min` channel-1 probe in `src/heightmap_verify.rs`
with `report: true` and count floor normals with `y < -0.5` vs `y > 0.5`. If the
downward ones form thin wall-shaped regions rather than noise, it holds. If they
are scattered, fall back to an ortho stencil pass over the cap plane, writing
`solid` — the plumbing for that already exists.

Whatever the outcome, `Heightmap::solid` is the channel to write, and the
gating in `shade_cascades.wgsl` already consumes it.

## Known limits

One split and one floor/ceiling interval cannot represent a storey slab *and*
the storey above it — the planned fix is artist-controlled up/down frustums, up
to three, with persistent probe hits and periodic regather. Don't try to solve
that inside the single-interval design.
