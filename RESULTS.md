Starting with the reference check I promised: I want to confirm the interval scaling and merge rule from a published 2D implementation before writing code that depends on them.

That confirms the scaling: probe spacing doubles per cascade, while ray count and interval length both grow 4x. Now the merge details:

That page describes the ordinary merge, not the fix. Osborne's paper covers the bilinear fix explicitly:

Now the Rust side: buffers, input handling, dispatches, and the error readback.

Builds. Running it for 40 seconds to watch the error readout converge:

The pipe swallowed the output when the timeout fired, so I'll log to a file instead:

Now the screenshots:

Quick rerun to confirm the new defaults and that the warning is gone:

The 2D testbed is built and working, and it gives you a number to tune against. It runs the cascades and a slow brute-force ray tracer (the reference) on the same scene with the same trace code. Every 2 seconds it logs the relative RMSE between them, at about 35 fps on your Iris Xe. Launch it with `cargo run --bin rc2d`; plain `cargo run` still starts the 3D app.

Controls: `1` / `2` / `3` show cascades / reference / difference, `B` toggles the bilinear fix, `Up` / `Down` doubles or halves `b`, and `R` restarts the reference.

**What I checked the design against.** The scaling rules (probe spacing doubles per cascade, ray count and interval length ×4), the merge rule, and the bilinear weights follow [m4xc's Fundamentals of RC](https://m4xc.dev/articles/fundamental-rc/). The bilinear fix follows Osborne & Sannikov ([arXiv:2408.14425](https://arxiv.org/html/2408.14425v1)): 4 traces per ray, one ending where each neighbouring probe's interval starts. The shader header records these choices so they're written down, not implied.

**Results** (reference converged past 40k rays per pixel):

| `b` | plain merge | bilinear fix |
|---|---|---|
| 0.25 | 8.54% | 6.37% |
| **0.5** | **5.63%** | **4.19%** |
| 1 | 6.42% | 4.93% |
| 2 | 9.94% | 8.42% |

- **The testbed catches the known failures.** With the plain merge, warm light leaks through the thin wall and there are rings at cascade boundaries. The bilinear fix removes the leak and most of the ringing.
- **What's left with the fix** is streaking along shadow edges, from each cascade's limited angular resolution, and a small ring near each light.
- **`b = 0.5` is the best setting in both modes:** cascade 0's rays reach 2 px, twice the probe spacing. That's one scene at one resolution, so it's a starting point, not a rule.
- **New defaults:** I set the bilinear fix on and `b = 0.5`. The table is in the doc comment at the top of `src/bin/rc2d.rs`.

**Also changed:**
- The error log now says "no data yet" instead of a fake 0% while the shaders compile.
- A duplicate-despawn warning from the readback is fixed.
- `Cargo.toml` now has `default-run = "s_rc"`.

**Limitations:**
- Scene lighting has no bounces; the scene only has emitters and black occluders.
- The reference and cascades share one trace function, so both inherit its limits: a 256-step cap and a 0.05 px hit threshold. The comparison isolates the cascade approximation, but it doesn't check the tracer itself.

Everything was tested by launching the real app with scripted key presses and screenshots. The temporary scripts are removed.

**This isn't committed yet.** Should I commit and push it to `moragtong/s_rc`, the same as last time? After that, the natural next test is a moving light, to see how the error behaves when the scene changes.
