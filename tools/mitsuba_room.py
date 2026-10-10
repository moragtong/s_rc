"""Render the checked-in Mitsuba version of `room.glb`.

Usage:
    tools/.venv/bin/python tools/mitsuba_room.py [spp] [--bounces N] [--white] \
        [--sun azimuth,elevation] [--cam ox,oy,oz,tx,ty,tz]

`--sun` is degrees: azimuth about +Y from +Z, elevation above the horizon.
`--cam` is origin and eye-space target, same as `S_RC_CAM`.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")
ROOT = Path.cwd()
OUT = ROOT / "target/mitsuba_room"
OBJ_DIR = ROOT / "assets/models/room_obj"
MANIFEST = OBJ_DIR / "manifest.json"
GLB = ROOT / "assets/models/room.glb"


def export_obj_if_needed():
    if MANIFEST.exists() and MANIFEST.stat().st_mtime > GLB.stat().st_mtime:
        return
    import subprocess
    import sys

    subprocess.run([sys.executable, str(ROOT / "tools/export_room_obj.py")], check=True)


def mesh_shapes(white=False):
    manifest = json.loads(MANIFEST.read_text())
    shapes = {}
    for rel, albedo in manifest.items():
        shapes[Path(rel).stem] = {
            "type": "obj",
            "filename": str(ROOT / rel),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [1.0, 1.0, 1.0] if white else albedo,
                },
            },
        }
    return shapes


def sun_direction(azimuth_deg, elevation_deg):
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    toward = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    return -toward  # Mitsuba directional: direction of travel


def scene_dict(sun_dir, eye, target, width, height, bounces, white=False, integrator=None):
    # Same exposure class as the app's default camera: default EV100_BLENDER
    # with the default 10000 lux directional light.  This gives the directional
    # light's pre-tonemap irradiance.  It is not meant to be exact physical
    # exposure; it just keeps the Mitsuba scene in the same rough units as the
    # app's HDR frame.
    irradiance = 10000.0 * (2.0 ** (-9.7)) / 1.2
    scene = {
        "type": "scene",
        "integrator": integrator
        or {"type": "path", "max_depth": bounces + 2},
        "sensor": {
            "type": "perspective",
            "fov": 45.0,
            "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=eye.tolist(), target=target.tolist(), up=[0.0, 1.0, 0.0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "rfilter": {"type": "box"},
                "pixel_format": "rgb",
            },
        },
        "sun": {
            "type": "directional",
            "direction": sun_dir.tolist(),
            "irradiance": {"type": "rgb", "value": [irradiance] * 3},
        },
        "sky": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": sky_radiance(irradiance).tolist()},
        },
    }
    scene.update(mesh_shapes(white=white))
    return scene


def sky_radiance(sun_irradiance):
    # Mirrors `sky_color` in src/cascades2_5d/mod.rs: rays that hit nothing
    # read as an overcast-blue environment, not as black.
    try:
        value = tuple(
            float(x.strip())
            for x in __import__("os").environ["S_RC_SKY"].split(",")
        )
        if len(value) >= 3:
            return np.array(value[:3], dtype=np.float32)
    except Exception:
        pass
    return np.array(
        [sun_irradiance * 0.05, sun_irradiance * 0.065, sun_irradiance * 0.11],
        dtype=np.float32,
    )


def write(image, name):
    OUT.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(image.astype(np.float32)).write(str(OUT / f"{name}.exr"))
    tonemapped = 1.0 - np.exp(-np.maximum(image, 0) * 2.0)
    mi.Bitmap(tonemapped.astype(np.float32)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    ).write(str(OUT / f"{name}.png"))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("spp", nargs="?", type=int, default=256)
    p.add_argument("--bounces", type=int, default=1)
    p.add_argument("--white", action="store_true")
    p.add_argument("--sun", default="45,54.7")
    p.add_argument("--cam", default="-9.0,5.0,5.0,0.0,1.2,0.0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args()

    export_obj_if_needed()
    sun_az, sun_el = [float(x) for x in a.sun.split(",")]
    cam = [float(x) for x in a.cam.split(",")]
    if len(cam) != 6:
        raise SystemExit("--cam needs ox,oy,oz,tx,ty,tz")
    eye = np.array(cam[:3], dtype=np.float64)
    target = np.array(cam[3:], dtype=np.float64)
    sun_dir = sun_direction(sun_az, sun_el)

    depth_scene = mi.load_dict(
        scene_dict(
            sun_dir,
            eye,
            target,
            a.width,
            a.height,
            0,
            white=a.white,
            integrator={"type": "aov", "aovs": "dd.y:depth"},
        )
    )
    depth = np.array(mi.render(depth_scene, spp=1))
    OUT.mkdir(parents=True, exist_ok=True)
    mi.Bitmap(depth[..., 0] if depth.ndim == 3 else depth).write(str(OUT / "depth.exr"))

    renders = {}
    for b in sorted({0, a.bounces}):
        scene = mi.load_dict(
            scene_dict(sun_dir, eye, target, a.width, a.height, b, white=a.white)
        )
        renders[b] = np.array(mi.render(scene, spp=a.spp))
        write(renders[b], f"b{b}")
    if a.bounces > 0:
        write(renders[a.bounces] - renders[0], f"indirect_b{a.bounces}")
    print(f"wrote {OUT}/b0.exr, b{a.bounces}.exr ({a.spp} spp)")


if __name__ == "__main__":
    main()
