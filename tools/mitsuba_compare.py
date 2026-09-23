"""Compares the app's cascade lighting against Mitsuba, pixel by pixel.

Inputs (from the same camera): the app's radiance export (target/rc3d/
radiance.f32, written with X in the SDF view's lighting mode) and Mitsuba's
white-albedo direct render (tools/mitsuba_render.py SPP --white). Both are
then irradiance / pi at each pixel's surface.

Reports relative RMSE and mean bias over pixels both sides light, pixels the
cascades leave black (no live probe: a coverage gap, not an accuracy
error), and writes a side-by-side / error image.

  tools/.venv/bin/python tools/mitsuba_compare.py [radiance.f32] [mitsuba.exr] [out_prefix]
"""

import sys
from pathlib import Path

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")
EXPORT = Path("target/rc3d")


def main():
    ours_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPORT / "radiance.f32"
    ref_path = Path(sys.argv[2]) if len(sys.argv) > 2 else EXPORT / "mitsuba_direct_white.exr"
    prefix = sys.argv[3] if len(sys.argv) > 3 else "compare"

    ref = np.array(mi.Bitmap(str(ref_path)))[..., :3].astype(np.float64)
    h, w = ref.shape[:2]
    ours = np.fromfile(ours_path, dtype="<f4").reshape(h, w, 4)[..., :3].astype(np.float64)

    lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])
    ref_lit = lum(ref) > 1e-4
    ours_lit = lum(ours) > 0
    both = ref_lit & ours_lit
    err = ours - ref
    rel_rmse = np.sqrt((err[both] ** 2).sum() / (ref[both] ** 2).sum())
    bias = lum(ours[both]).sum() / lum(ref[both]).sum() - 1
    print(f"lit in both: {both.sum()} px; black in ours but lit in Mitsuba (no live probe): "
          f"{(ref_lit & ~ours_lit).sum()} px")
    print(f"relative RMSE {100 * rel_rmse:.1f}%, brightness {100 * bias:+.1f}% vs Mitsuba")
    # Per-pixel relative error distribution (luminance), ignoring very dark pixels.
    bright = both & (lum(ref) > 0.05 * lum(ref[both]).mean())
    per_px = np.abs(lum(ours) - lum(ref))[bright] / lum(ref)[bright]
    print(f"per-pixel |relative error| (non-dark pixels): median {100 * np.median(per_px):.1f}%, "
          f"90% {100 * np.percentile(per_px, 90):.1f}%")

    tone = lambda x: 1 - np.exp(-2 * x)
    diff = np.clip(np.abs(lum(err)) / (lum(ref).mean() + 1e-6), 0, 1)
    row = np.concatenate([tone(ours), tone(ref), np.stack([diff, diff * 0.3, 0 * diff], -1)], axis=1)
    mi.Bitmap(row.astype(np.float32)).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    ).write(str(EXPORT / f"{prefix}.png"))
    print(f"wrote {EXPORT / (prefix + '.png')}  (ours | Mitsuba | |error|)")


if __name__ == "__main__":
    main()
