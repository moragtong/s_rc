"""Where the SDF grid is wrong: rebuilds the stored field on the CPU exactly
as the GPU composes it (per-mesh baked grids from target/sdf_cache, sampled
like clipmap_compose.wgsl's sample_instance (outside a grid, the distance
to the mesh's bounds, here the box itself), min over instances at voxel
centers, rounded to f16), then filters it trilinearly like the samplers
tracing it do, and compares both with the exact box distances.

  tools/.venv/bin/python tools/sdf_error.py [scene.json]
"""
import json, sys
from pathlib import Path
import numpy as np

VOXEL = 0.125          # grid voxel (src/clipmap.rs)
BAKE_VOXEL = 0.0625    # per-mesh bake
PADDING = 3

def load_grids():
    grids = []
    for f in Path("target/sdf_cache").glob("*.sdf"):
        b = f.read_bytes()
        if b[:4] != b"SDF1":
            continue
        dims = np.frombuffer(b[4:16], "<u4")
        mn = np.frombuffer(b[16:28], "<f4"); vox = np.frombuffer(b[28:32], "<f4")[0]
        if abs(vox - BAKE_VOXEL) > 1e-6:
            continue
        vals = np.frombuffer(b[32:], "<f4").reshape(dims[2], dims[1], dims[0])  # z, y, x
        grids.append((mn.astype(np.float64), float(vox), dims, vals))
    return grids

def sample_instance(g, p, half):
    """clipmap_compose.wgsl sample_instance, vectorised; p (N,3) local."""
    mn, v, dims, vals = g
    lo = mn + 0.5 * v; hi = mn + (dims - 0.5) * v
    outside = np.any((p < lo) | (p > hi), axis=1)
    mesh_lo, mesh_hi = -half, half
    bounds = np.linalg.norm(np.maximum(np.maximum(mesh_lo - p, p - mesh_hi), 0), axis=1)
    q = np.clip(p, lo, hi)
    gq = (q - mn) / v - 0.5
    c = np.floor(gq).astype(int); f = gq - c
    def val(dx, dy, dz):
        i = np.clip(c + [dx, dy, dz], 0, dims - 1)
        return vals[i[:, 2], i[:, 1], i[:, 0]]
    x00 = val(0,0,0) * (1-f[:,0]) + val(1,0,0) * f[:,0]
    x10 = val(0,1,0) * (1-f[:,0]) + val(1,1,0) * f[:,0]
    x01 = val(0,0,1) * (1-f[:,0]) + val(1,0,1) * f[:,0]
    x11 = val(0,1,1) * (1-f[:,0]) + val(1,1,1) * f[:,0]
    d = (x00 * (1-f[:,1]) + x10 * f[:,1]) * (1-f[:,2]) + (x01 * (1-f[:,1]) + x11 * f[:,1]) * f[:,2]
    return np.where(outside, bounds, d)

def box_sdf(p, c, h):
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)

def main():
    scene = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "target/rc3d/full_scene.json"))
    boxes = [(np.array(b["center"]), np.array(b["half_size"])) for b in scene["boxes"]]
    grids = load_grids()
    # Match each box to its baked grid by extent (room boxes are axis-aligned,
    # baked in local space around the origin).
    inst = []
    for c, h in boxes:
        ext = [g for g in grids if np.allclose(g[0], -h - PADDING * BAKE_VOXEL, atol=0.07)]
        if not ext:
            sys.exit(f"no cached grid for box half {h}")
        inst.append((c, h, ext[0]))
    print(f"{len(inst)} boxes matched to baked grids")

    # Composed field at voxel centers over the room (+ margin).
    lo = np.array([-4.75, -0.5, -4.75]); hi = np.array([4.75, 3.75, 4.75])
    ks = [np.arange(np.floor(lo[a] / VOXEL), np.ceil(hi[a] / VOXEL)) for a in range(3)]
    kx, ky, kz = np.meshgrid(*ks, indexing="ij")
    centers = (np.stack([kx, ky, kz], -1).reshape(-1, 3) + 0.5) * VOXEL
    stored = np.min([sample_instance(g, centers - c, h) for c, h, g in inst], axis=0)
    stored16 = stored.astype(np.float16).astype(np.float64)
    exact_c = np.min([box_sdf(centers, c, h) for c, h, g in inst], axis=0)
    shape = kx.shape
    field = stored16.reshape(shape)
    k0 = np.array([k[0] for k in ks])

    def trilinear(p):
        g = p / VOXEL - 0.5 - k0
        c = np.floor(g).astype(int); f = g - c
        def val(dx, dy, dz):
            i = np.clip(c + [dx, dy, dz], 0, np.array(shape) - 1)
            return field[i[:, 0], i[:, 1], i[:, 2]]
        x00 = val(0,0,0) * (1-f[:,0]) + val(1,0,0) * f[:,0]
        x10 = val(0,1,0) * (1-f[:,0]) + val(1,1,0) * f[:,0]
        x01 = val(0,0,1) * (1-f[:,0]) + val(1,0,1) * f[:,0]
        x11 = val(0,1,1) * (1-f[:,0]) + val(1,1,1) * f[:,0]
        return (x00 * (1-f[:,1]) + x10 * f[:,1]) * (1-f[:,2]) + (x01 * (1-f[:,1]) + x11 * f[:,1]) * f[:,2]

    out = exact_c > 0
    err = stored16 - exact_c
    print(f"\nat voxel centers ({len(centers)}): outside the solid |error| median {np.median(np.abs(err[out]))*100:.2f} cm, "
          f"max {np.abs(err[out]).max()*100:.2f} cm; inside max overestimate {err[~out].max()*100:.2f} cm")

    # What rays see: the filtered field at fine sample points. Inside any box
    # (true solid), a filtered value above the hit threshold is an opening a
    # ray can pass through.
    rng = np.random.default_rng(1)
    pts = rng.uniform(lo, hi, size=(3_000_000, 3))
    f = trilinear(pts)
    exact = np.min([box_sdf(pts, c, h) for c, h, g in inst], axis=0)
    inside = exact < 0
    print(f"\nfiltered field at {len(pts)} random points:")
    print(f"  outside the solid: error median {np.median(np.abs(f - exact)[~inside])*100:.2f} cm, "
          f"95% {np.percentile(np.abs(f - exact)[~inside], 95)*100:.2f} cm, max over {((f - exact)[~inside]).max()*100:.2f} cm")
    for eps in [0.03125, 0.0156, 0.0]:
        open_ = inside & (f >= eps)
        print(f"  inside the solid but filtered distance >= {eps*100:.2f} cm (open to rays): {open_.sum()} points "
              f"({open_.sum()/inside.sum()*100:.3f}% of inside)")
        if open_.any() and eps > 0:
            p = pts[open_]
            # Which features: distance to the nearest two boxes' surfaces.
            d_all = np.stack([box_sdf(p, c, h) for c, h, g in inst], 1)
            order = np.argsort(d_all, 1)
            depth = -d_all[np.arange(len(p)), order[:, 0]]
            print(f"    depth inside the solid: median {np.median(depth)*100:.2f} cm, max {depth.max()*100:.2f} cm; "
                  f"filtered value max {f[open_].max()*100:.2f} cm")
            # Cluster by rounding to 25 cm to show where.
            cells, counts = np.unique(np.round(p / 0.25) * 0.25, axis=0, return_counts=True)
            top = np.argsort(counts)[::-1][:12]
            print("    top locations (25 cm cells, count):", [(tuple(np.round(cells[i], 2)), int(counts[i])) for i in top])

if __name__ == "__main__":
    main()

def overestimates():
    """Where the stored value exceeds the true distance outside the solid
    (sphere tracing steps too far there), and by how much per instance."""
    scene = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "target/rc3d/full_scene.json"))
    boxes = [(np.array(b["center"]), np.array(b["half_size"])) for b in scene["boxes"]]
    grids = load_grids()
    inst = [(c, h, [g for g in grids if np.allclose(g[0], -h - PADDING * BAKE_VOXEL, atol=0.07)][0]) for c, h in boxes]
    lo = np.array([-4.75, -0.5, -4.75]); hi = np.array([4.75, 3.75, 4.75])
    ks = [np.arange(np.floor(lo[a] / VOXEL), np.ceil(hi[a] / VOXEL)) for a in range(3)]
    kx, ky, kz = np.meshgrid(*ks, indexing="ij")
    centers = (np.stack([kx, ky, kz], -1).reshape(-1, 3) + 0.5) * VOXEL
    per = np.stack([sample_instance(g, centers - c, h) for c, h, g in inst], 1)
    exact_per = np.stack([box_sdf(centers, c, h) for c, h, g in inst], 1)
    stored = per.min(1); exact = exact_per.min(1)
    over = stored - exact
    out = exact > 0
    print("\noverestimate at voxel centers outside the solid:")
    for t in [0.01, 0.03, 0.06]:
        print(f"  > {t*100:.0f} cm: {(out & (over > t)).sum()} voxels")
    worst = np.argsort(np.where(out, over, -1))[::-1][:8]
    for i in worst:
        nearest = int(np.argmin(exact_per[i]))
        own_err = per[i, nearest] - exact_per[i, nearest]
        print(f"  at {np.round(centers[i], 3)}: stored {stored[i]*100:.1f} cm, true {exact[i]*100:.1f} cm; "
              f"nearest box {nearest} (half {boxes[nearest][1]}) sampled {per[i, nearest]*100:.1f} cm (its error {own_err*100:+.1f} cm)")

overestimates()
