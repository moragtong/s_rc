"""Which cascade rays change outcome with the hit threshold: marches one
cascade's segments (as cascades3d.wgsl builds them) against the SDF field
rebuilt on the CPU (tools/sdf_error.py's compose), at two thresholds, and
reports where the segments that hit at the larger one but miss at the
smaller one pass closest to the geometry.

  tools/.venv/bin/python tools/threshold_rays.py [voxel] [cascade] [--inside]
"""
import json, sys
import numpy as np
sys.path.insert(0, "tools")
import sdf_error as se

VOXEL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0625
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SPACING0, OCTA0, INTERVAL0 = 0.5, 4, 1.5

def oct_decode(uv):
    f = uv * 2 - 1
    n = np.stack([f[:, 0], f[:, 1], 1 - np.abs(f[:, 0]) - np.abs(f[:, 1])], 1)
    t = np.maximum(-n[:, 2], 0)
    n[:, 0] += np.where(n[:, 0] >= 0, -t, t)
    n[:, 1] += np.where(n[:, 1] >= 0, -t, t)
    return n / np.linalg.norm(n, axis=1, keepdims=True)

def main():
    se.VOXEL, se.BAKE_VOXEL = VOXEL, VOXEL / 2
    scene = json.load(open("target/rc3d/full_scene.json"))
    boxes = [(np.array(b["center"]), np.array(b["half_size"])) for b in scene["boxes"]]
    grids = se.load_grids()
    inst = [(c, h, [g for g in grids if np.allclose(g[0], -h - se.PADDING * se.BAKE_VOXEL, atol=0.05)][0]) for c, h in boxes]
    lo = np.array([-6.0, -0.5, -6.0]); hi = np.array([6.0, 4.5, 6.0])
    ks = [np.arange(np.floor(lo[a] / VOXEL), np.ceil(hi[a] / VOXEL)) for a in range(3)]
    kx, ky, kz = np.meshgrid(*ks, indexing="ij")
    centers = (np.stack([kx, ky, kz], -1).reshape(-1, 3) + 0.5) * VOXEL
    field = np.min([se.sample_instance(g, centers - c, h) for c, h, g in inst], axis=0).astype(np.float16).astype(np.float64).reshape(kx.shape)
    k0 = np.array([k[0] for k in ks]); shape = np.array(kx.shape)
    def sdf(p):
        g = p / VOXEL - 0.5 - k0
        c = np.floor(g).astype(int); f = g - c
        def val(dx, dy, dz):
            i = np.clip(c + [dx, dy, dz], 0, shape - 1)
            return field[i[:, 0], i[:, 1], i[:, 2]]
        x00 = val(0,0,0) * (1-f[:,0]) + val(1,0,0) * f[:,0]
        x10 = val(0,1,0) * (1-f[:,0]) + val(1,1,0) * f[:,0]
        x01 = val(0,0,1) * (1-f[:,0]) + val(1,0,1) * f[:,0]
        x11 = val(0,1,1) * (1-f[:,0]) + val(1,1,1) * f[:,0]
        return (x00 * (1-f[:,1]) + x10 * f[:,1]) * (1-f[:,2]) + (x01 * (1-f[:,1]) + x11 * f[:,1]) * f[:,2]

    s = SPACING0 * 2 ** N; side = OCTA0 << N; end = INTERVAL0 * 2 ** N
    start = end / 2 if N > 0 else 0.0
    cs = s * 2
    # Probes in and around the room, in empty space.
    pk = [np.arange(np.floor(-5 / s), np.ceil(5 / s)) for _ in range(3)]
    pk[1] = np.arange(0, np.ceil(3.5 / s))
    px, py, pz = np.meshgrid(*pk, indexing="ij")
    probes = (np.stack([px, py, pz], -1).reshape(-1, 3) + 0.5) * s
    probes = probes[sdf(probes) > 0]
    if "--inside" in sys.argv:
        probes = probes[(np.abs(probes[:, 0]) < 3.9) & (np.abs(probes[:, 2]) < 3.9) & (probes[:, 1] < 3.0)]
    cells = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), -1).reshape(-1, 2)
    dirs = oct_decode((cells + 0.5) / side)
    corners = np.array([[k & 1, (k >> 1) & 1, k >> 2] for k in range(8)])
    P = np.repeat(probes, len(dirs) * 8, 0)
    D = np.tile(np.repeat(dirs, 8, 0), (len(probes), 1))
    K = np.tile(corners, (len(probes) * len(dirs), 1))
    base = np.floor(P / cs - 0.5)
    child = (base + K + 0.5) * cs
    A = P + D * start
    B = child + D * end
    delta = B - A; length = np.linalg.norm(delta, axis=1); dirn = delta / length[:, None]
    print(f"voxel {VOXEL*100:.2f} cm, cascade {N}: {len(probes)} probes, {len(A)} segments")

    def march(eps):
        t = np.zeros(len(A)); hit = np.zeros(len(A), bool); done = np.zeros(len(A), bool)
        closest = np.full(len(A), 1e9); closest_p = A.copy()
        for _ in range(256):
            live = ~done
            if not live.any():
                break
            p = A[live] + dirn[live] * t[live, None]
            d = sdf(p)
            idx = np.nonzero(live)[0]
            better = d < closest[idx]
            closest[idx[better]] = d[better]; closest_p[idx[better]] = p[better]
            h = d < eps
            hit[idx[h]] = True; done[idx[h]] = True
            t[idx[~h]] += np.maximum(d[~h], 0.01)
            done[idx[t[idx] >= length[idx]]] = True
        return hit, closest, closest_p

    hit_small, _, _ = march(0.0156)
    hit_large, closest, closest_p = march(0.03125)
    flips = hit_large & ~hit_small
    print(f"segments hitting at 3.1 cm but passing at 1.6 cm: {flips.sum()} ({flips.mean()*100:.2f}%)")
    if flips.any():
        cp = closest_p[flips]
        # Distance to the nearest box edge line vs face: classify.
        cells_, counts = np.unique(np.round(cp / 0.25) * 0.25, axis=0, return_counts=True)
        top = np.argsort(counts)[::-1][:15]
        print("  where they pass closest (25 cm cells, count):")
        for i in top:
            print(f"    {tuple(np.round(cells_[i], 2))}: {counts[i]}")
        # How many pass within 5 cm of a door jamb edge (x in [-4.1,-3.9], |z| = 0.6, y < 2.2)?
        jamb = (np.abs(np.abs(cp[:, 2]) - 0.6) < 0.08) & (cp[:, 0] > -4.2) & (cp[:, 0] < -3.8) & (cp[:, 1] < 2.3)
        lintel = (np.abs(cp[:, 1] - 2.2) < 0.08) & (cp[:, 0] > -4.2) & (cp[:, 0] < -3.8) & (np.abs(cp[:, 2]) < 0.7)
        print(f"  within 8 cm of a door jamb edge: {jamb.sum()}, of the lintel's lower edge: {lintel.sum()}, elsewhere: {(~jamb & ~lintel).sum()}")

main()
