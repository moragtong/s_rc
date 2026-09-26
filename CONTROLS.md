Run the testbed from the project folder, then press `S` once the window opens:

```
cargo run --bin rc2d
```

`S` switches between the two scenes, and switching restarts the reference.

Keys while in the room:

| key | what it does |
|---|---|
| `1` / `2` / `3` | cascade result / reference / difference view |
| left-drag | move the spotlight |
| `G` | number of bounces (0–4, default 2) |
| `B` | switch between the bilinear fix and the plain merge, to see the leak |
| `P`, `Up`/`Down`, `D` | probe spacing, cascade-0 interval `b`, per-direction levels |

A few things to expect:
- **It runs slowly, around 10 fps.** Most of that is the reference path tracer, which runs every frame even while you're looking at the cascade view.
- **The reference starts noisy.** The reference view (`2`) and the difference view (`3`) take a minute or so to settle. The cascade view (`1`) is final immediately.
- **The walls are only 2 px wide,** so their lit faces are easy to miss at normal size. A screen magnifier helps.
- **The terminal logs error figures every 2 seconds.** In the room scene that includes the in-room and wall-face figures.
