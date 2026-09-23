"""Mitsuba 3 lighting reference for the 3D room, from the camera the app
exported (target/rc3d/scene.json, written with X).

Direct lighting only for now (path integrator, max_depth = 2: camera ->
surface -> light), Lambertian boxes, each light an emissive sphere with
uniform radiance. The spotlight's cone is ignored (isotropic), matching the
app's first 3D cascade milestone; Mitsuba has no built-in emitter for a
sphere with a cone.

Writes target/rc3d/mitsuba_direct.exr (linear radiance) and a tonemapped
PNG. Also writes mitsuba_depth.exr so the camera convention can be checked
against the app's traced depth.

With --white, every surface is albedo 1, so each pixel is exactly the
irradiance at its surface divided by pi: that isolates the lighting from the
albedo source, which the SDF doesn't carry yet (ARCHITECTURE.md's open
problem). Outputs then get a _white suffix.

Run with the project venv:
  tools/.venv/bin/python tools/mitsuba_render.py [spp] [--white]
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
    for i, light in enumerate(scene_json["lights"]):
        d[f"light{i}"] = {
            "type": "sphere",
            "center": light["center"],
            "radius": light["radius"],
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": light["radiance"]}},
        }
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    white = "--white" in sys.argv
    suffix = "_white" if white else ""
    spp = int(args[0]) if args else 256
    scene_json = json.loads((EXPORT / "scene.json").read_text())

    depth_scene = mi.load_dict(scene_dict(scene_json, {"type": "aov", "aovs": "dd.y:depth"}))
    depth = np.array(mi.render(depth_scene, spp=1))[..., 0]
    mi.Bitmap(depth).write(str(EXPORT / "mitsuba_depth.exr"))

    scene = mi.load_dict(scene_dict(scene_json, {"type": "path", "max_depth": 2}, white))
    image = mi.render(scene, spp=spp)
    mi.Bitmap(image).write(str(EXPORT / f"mitsuba_direct{suffix}.exr"))
    rgb = np.array(image)
    tonemapped = 1.0 - np.exp(-rgb * 2.0)
    mi.Bitmap(tonemapped.astype(np.float32)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    ).write(str(EXPORT / f"mitsuba_direct{suffix}.png"))
    print(f"wrote {EXPORT}/mitsuba_direct{suffix}.exr/.png ({spp} spp) and mitsuba_depth.exr")


if __name__ == "__main__":
    main()
