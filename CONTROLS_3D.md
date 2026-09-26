I'm writing the GPU clipmap. The Rust side is drafted, but I left some leftover junk in the upload function, which I'm cleaning up now before writing the compose shader:

The linker crashed with a bus error, not a code error. With `lld`, that usually means a disk or memory space problem. Checking:

That freed 9.3 GB; there were only 5 executables, but at 1.8 GB each. Now 12 GB free. Changing the debug-info settings:

Now the Rust side of the view: the shared uniform, a full-window UI image toggled with `V` (`N` cycles the display mode), and the `X` export.

Now `main.rs`: the room replaces the placeholder scene, and the camera gets the fly controller and becomes the one the clipmap follows:

The earlier build hadn't actually succeeded; the error was cut off by my output filter. The fix is the system set I was going to add anyway, which also keeps the compose internals private:

It builds. Rerunning the same script:

Now the test reproduces the crash against the old code and passes with the fix. Rerunning the 3D app:

Shaders loaded, and 313k of 360k pixels hit (the rest is sky). The screenshots:

Inside the room: **no leaks at all**, every hit is in tier 0, median position error 6.4 mm, max 4.8 cm. The clipmap is validated. Next I'm cleaning up compiler warnings, then committing and pushing.

Only the known warning remains, and all tests pass. I'll pin the Python tool's dependencies so the venv can be recreated, then commit and push without your notes files:

A stray `tools/__pycache__/` came from my diagnostic imports. I'll delete it and ignore it:

The 3D clipmap works. It's built on the GPU from `parry3d`-baked SDFs, scrolls with the camera, and its geometry agrees with Mitsuba to within millimetres nearby. Pushed as `486438b` (plus a one-line `.gitignore` fix).

**What's in the main app now:**
- **Scene:** an analytic 3D version of the room: 20 cm walls, ceiling, one doorway, a spotlight outside aimed beside it, three more lights, and a fly camera. One definition feeds the Bevy meshes, the SDF bake and the Mitsuba export.
- **Bake:** each mesh gets its own SDF grid through `parry3d`, cached to disk. Tests check a baked box against the exact box distance: under 0.1 mm everywhere, including inside.
- **Clipmap:** 3 tiers of 128³ voxels at 12.5 cm, 50 cm and 2 m, centred on the camera. When you move, only newly revealed slabs are rebuilt. A test checks those slabs cover exactly the new voxels.

**Checked against Mitsuba.** It fires the identical pixel rays at Mitsuba's own copy of the boxes, then reports rays that pass through geometry Mitsuba hits (leaks), and how far each clipmap hit is from the true surface:

| | leaks | nearest tier (12.5 cm): error | next tier (50 cm): error |
|---|---|---|---|
| outside, looking through the doorway | 987 px (0.27%) | median 4.8 mm, max 3.9 cm | all within one 50 cm voxel |
| inside the room | 0 | median 6.4 mm, max 4.8 cm | — |

The 987 remaining misses are rays grazing sharp outer edges, which the smoothed SDF rounds off slightly. They clip the true geometry by 3 mm on average. None passes through a wall.

**Scrolling is exact.** After walking 7.5 m forward and 3 m sideways, the scrolled clipmap traces depth bit-for-bit identical to one rebuilt from scratch.

**A design choice for you to review:**
- **The problem:** the first comparison showed rays **passing through the room's far wall**. At that distance the ray samples the 50 cm tier, and a 20 cm wall falls between its sample points, so it vanishes. For lighting, that's a leak.
- **The fix:** the coarser tiers now count a hit within half a voxel of a surface, which provably catches thin walls. That cut the misses from 19,359 pixels to 987.
- **The cost:** rays skimming a floor or ceiling at a shallow angle stop early, up to 1–1.5 m along the ray, while landing within 25 cm of a real surface.
- It's the trade ARCHITECTURE.md asks for: lost light, never leaks. It will also govern how lighting rays trace later.

**Not done yet:**
- Only geometry is compared with Mitsuba so far. Comparing lighting needs the 3D cascades, which are next.
- Lights aren't in the SDF yet.
- Beyond each mesh's grid, the distance is estimated in a way that can overshoot. It hasn't caused visible errors, and the bake's padding keeps it rare.
- The inside/outside test assumes closed meshes, fine for boxes but not Sponza.

**Disk space:** the build failed with your disk 98% full. Each debug binary was 1.8 GB, mostly debug info. I deleted 9.3 GB of stale test builds from `target/` and cut debug info for dependencies (our own code keeps line numbers). Binaries are now 205 MB, but the disk is still at 97% with 13 GB free.

**To try it:** `cargo run`. Right-drag looks, `WASD`/`Q`/`E` move, `V` shows the SDF view, `N` cycles what it shows, `K` rebuilds the clipmap, and `X` exports for `tools/.venv/bin/python tools/mitsuba_depth.py`.

Next is the 3D cascades on top of the clipmap, with Mitsuba lighting renders as the reference, then Sponza.
