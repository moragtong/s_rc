"""Checks the SDF clipmap's geometry against Mitsuba 3's ray tracer.

Reads target/rc3d/scene.json and target/rc3d/sdf_depth.f32 (exported with X
in the app), casts the exact same pixel-center rays through Mitsuba's own
intersection of the analytic boxes, and compares:

- hit/miss agreement: a pixel Mitsuba hits but the clipmap misses is a ray
  that went through geometry, i.e. a leak;
- position error: how far each clipmap hit point is from the true surface
  (the exact box SDF there), per clipmap tier, against that tier's voxel.
  That's what matters for lighting. Depth along the ray is also reported,
  but it inflates position error by 1/sin(angle) at grazing incidence.

Run with the project venv:  tools/.venv/bin/python tools/mitsuba_depth.py
"""

import json
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np

mi.set_variant("llvm_ad_rgb")

EXPORT = Path("target/rc3d")
# The clipmap's finest voxel, for putting errors in scale.
VOXEL = 0.125


def load_scene(scene_json):
    shapes = {"type": "scene"}
    for i, b in enumerate(scene_json["boxes"]):
        shapes[f"box{i}"] = {
            "type": "cube",  # [-1, 1]^3
            "to_world": mi.ScalarTransform4f().translate(b["center"]).scale(b["half_size"]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": b["albedo"]}},
        }
    return mi.load_dict(shapes)


def exact_sdf(boxes, points):
    d = np.full(len(points), np.inf)
    for b in boxes:
        q = np.abs(points - np.array(b["center"])) - np.array(b["half_size"])
        d = np.minimum(d, np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0))
    return d


def tier_of(clipmap, points):
    """Finest tier whose window (minus the tracer's 1.5 voxel margin) holds each point."""
    tier = np.full(len(points), len(clipmap))
    for t in reversed(range(len(clipmap))):
        c = clipmap[t]
        lo = (np.array(c["origin"]) + 1.5) * c["voxel"]
        hi = (np.array(c["origin"]) + c["resolution"] - 1.5) * c["voxel"]
        inside = np.all((points > lo) & (points < hi), axis=-1)
        tier[inside] = t
    return tier


def pixel_rays(camera):
    """The app's rays: forward + ndc.x * right * tan * aspect + ndc.y * up * tan."""
    w, h = camera["width"], camera["height"]
    forward, up = np.array(camera["forward"]), np.array(camera["up"])
    right = np.cross(forward, up)
    tan = np.tan(np.radians(camera["fov_y_degrees"]) / 2)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    ndc_x = (xs + 0.5) / w * 2 - 1
    ndc_y = 1 - (ys + 0.5) / h * 2
    dirs = forward + (ndc_x * tan * w / h)[..., None] * right + (ndc_y * tan)[..., None] * up
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs.reshape(-1, 3)


def main():
    camera_scene = json.loads((EXPORT / "scene.json").read_text())
    camera = camera_scene["camera"]
    w, h = camera["width"], camera["height"]
    sdf = np.fromfile(EXPORT / "sdf_depth.f32", dtype="<f4")
    if sdf.size != w * h:
        sys.exit(f"sdf_depth.f32 has {sdf.size} values, expected {w * h}")

    scene = load_scene(camera_scene)
    dirs = pixel_rays(camera)
    origin = np.array(camera["position"])
    ray = mi.Ray3f(
        o=mi.Point3f(*[np.full(len(dirs), origin[k], dtype=np.float32) for k in range(3)]),
        d=mi.Vector3f(*[dirs[:, k].astype(np.float32) for k in range(3)]),
    )
    si = scene.ray_intersect(ray)
    ref = np.where(np.array(si.is_valid()), np.array(si.t), -1.0)

    sdf_hit, ref_hit = sdf > 0, ref > 0
    both = sdf_hit & ref_hit
    err = np.abs(sdf[both] - ref[both])
    print(f"pixels: {w * h}, both hit: {both.sum()}, "
          f"hit in SDF only: {(sdf_hit & ~ref_hit).sum()}, "
          f"hit in Mitsuba only (leaks): {(ref_hit & ~sdf_hit).sum()}")
    if both.any():
        pct = lambda p: np.percentile(err, p)
        print(f"|depth error| where both hit (m): median {pct(50):.4f}, 95% {pct(95):.4f}, "
              f"99% {pct(99):.4f}, max {err.max():.4f}")

    points = origin + dirs[sdf_hit] * sdf[sdf_hit][:, None]
    position = np.abs(exact_sdf(camera_scene["boxes"], points))
    tiers = tier_of(camera_scene["clipmap"], points)
    print("position error of clipmap hits (distance to the true surface):")
    for t, c in enumerate(camera_scene["clipmap"]):
        m = tiers == t
        if not m.any():
            continue
        e, v = position[m], c["voxel"]
        print(f"  tier {t} ({v} m voxels): {m.sum()} hits, median {np.median(e):.4f} m, "
              f"99% {np.percentile(e, 99):.4f} m, max {e.max():.4f} m; "
              f"within half a voxel: {100 * (e <= v / 2).mean():.2f}%")

    # Error image: green = within 1 cm, yellow -> red up to one voxel and
    # beyond, blue = hit/miss disagreement, black = both miss.
    img = np.zeros((w * h, 3), dtype=np.float32)
    e = np.zeros(w * h)
    e[both] = err
    frac = np.clip(e / VOXEL, 0, 1)
    img[both] = np.stack([frac[both], 1 - 0.7 * frac[both], np.zeros(both.sum())], -1)
    img[sdf_hit != ref_hit] = [0.2, 0.4, 1.0]
    mi.Bitmap(img.reshape(h, w, 3)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    ).write(str(EXPORT / "depth_error.png"))
    print(f"wrote {EXPORT / 'depth_error.png'}")


if __name__ == "__main__":
    main()
