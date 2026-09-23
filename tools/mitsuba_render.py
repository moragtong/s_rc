"""Mitsuba 3 lighting reference for the 3D room, from the camera the app
exported (target/rc3d/scene.json, written with X).

Lambertian boxes lit by the sun, a directional light (Mitsuba's is a delta
light, so its shadows are hard; the app's sun has the real 0.27 degree
radius, a negligible penumbra at room scale). No sky.

  tools/.venv/bin/python tools/mitsuba_render.py [spp] [--bounces N] [--white]

--bounces N (default 1): path max_depth = N + 2, i.e. direct light plus N
diffuse bounces; the app's cascades carry one bounce. Also renders direct
only, and writes their difference, the indirect part, for comparing the
cascades alone. Outputs: mitsuba_b{N}.exr, mitsuba_b0.exr, mitsuba_indirect_b{N}.exr
(+ tonemapped PNGs), and mitsuba_depth.exr for checking the camera.
--white sets every albedo to 1.
"""

import json
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np

mi.set_variant("llvm_ad_rgb")

EXPORT = Path("target/rc3d")


def scene_dict(scene_json, integrator, white=False):
    cam = scene_json["camera"]
    position = np.array(cam["position"])
    d = {
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": cam["fov_y_degrees"],
            "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=position.tolist(),
                target=(position + np.array(cam["forward"])).tolist(),
                up=cam["up"],
            ),
            "film": {
                "type": "hdrfilm",
                "width": cam["width"],
                "height": cam["height"],
                "rfilter": {"type": "box"},
                "pixel_format": "rgb",
            },
        },
    }
    for i, b in enumerate(scene_json["boxes"]):
        d[f"box{i}"] = {
            "type": "cube",
            "to_world": mi.ScalarTransform4f().translate(b["center"]).scale(b["half_size"]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0] if white else b["albedo"]}},
        }
    sun = scene_json["sun"]
    d["sun"] = {
        "type": "directional",
        "direction": (-np.array(sun["toward"])).tolist(),
        "irradiance": {"type": "rgb", "value": sun["irradiance"]},
    }
    return d


def write(image, name):
    mi.Bitmap(image.astype(np.float32)).write(str(EXPORT / f"{name}.exr"))
    tonemapped = 1.0 - np.exp(-np.maximum(image, 0) * 2.0)
    mi.Bitmap(tonemapped.astype(np.float32)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    ).write(str(EXPORT / f"{name}.png"))


def main():
    args = sys.argv[1:]
    white = "--white" in args
    bounces = int(args[args.index("--bounces") + 1]) if "--bounces" in args else 1
    numbers = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--bounces")]
    spp = int(numbers[0]) if numbers else 256
    scene_json = json.loads((EXPORT / "scene.json").read_text())

    depth_scene = mi.load_dict(scene_dict(scene_json, {"type": "aov", "aovs": "dd.y:depth"}))
    depth = np.array(mi.render(depth_scene, spp=1))
    mi.Bitmap(depth[..., 0] if depth.ndim == 3 else depth).write(str(EXPORT / "mitsuba_depth.exr"))

    renders = {}
    for b in sorted({0, bounces}):
        scene = mi.load_dict(scene_dict(scene_json, {"type": "path", "max_depth": b + 2}, white))
        renders[b] = np.array(mi.render(scene, spp=spp))
        write(renders[b], f"mitsuba_b{b}")
    if bounces > 0:
        write(renders[bounces] - renders[0], f"mitsuba_indirect_b{bounces}")
    print(f"wrote {EXPORT}/mitsuba_b0, mitsuba_b{bounces}, mitsuba_indirect_b{bounces} ({spp} spp)")


if __name__ == "__main__":
    main()
