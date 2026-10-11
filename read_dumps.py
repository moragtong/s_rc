#!/usr/bin/env python3
"""Read and visualize s_rc heightmap/winding dumps as PNG."""

import numpy as np
import os
import sys
from PIL import Image

def normalize_for_image(arr, vmin=None, vmax=None):
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return (normalized * 255).astype(np.uint8)

def main():
    if len(sys.argv) < 2:
        print("Usage: read_dumps.py <dump_dir> [output_dir]")
        print("  Dumps: w.bin/winding.bin, h.bin/height.bin, mm.bin/meta_min.bin, mf.bin/meta_max.bin")
        print("  Outputs: PNG files in current directory")
        return

    dump_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Winding
    for name in ["w.bin", "winding.bin"]:
        p = os.path.join(sys.argv[1], name)
        if os.path.exists(p):
            winding = np.fromfile(p, dtype=np.float32).reshape(512, 512)
            pos_mask = winding > 0
            neg_mask = winding < 0
            zero_mask = ~pos_mask & ~neg_mask
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            rgb[pos_mask, 0] = normalize_for_image(winding[pos_mask], 0, 1)
            rgb[neg_mask, 2] = normalize_for_image(-winding[neg_mask], 0, 1)
            rgb[zero_mask, 0] = 128
            rgb[zero_mask, 1] = 128
            rgb[zero_mask, 2] = 128
            Image.fromarray(rgb).save("winding.png")
            print(f"Saved winding to winding.png")
            solid = (np.abs(winding) > 0.5).sum()
            open_ = (np.abs(winding) < 0.1).sum()
            print(f"  Winding: solid={solid}, open={open_}")
            break

    # Height
    for name in ["h.bin", "height.bin"]:
        p = os.path.join(sys.argv[1], name)
        if os.path.exists(p):
            height = np.fromfile(p, dtype=np.float32).reshape(512, 512, 2)
            ceiling = height[:, :, 0]
            floor = height[:, :, 1]
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            rgb[:, :, 0] = normalize_for_image(np.clip(ceiling, -5, 5), -5, 5)
            rgb[:, :, 1] = normalize_for_image(np.clip(floor, -15, 5), -15, 5)
            rgb[:, :, 2] = normalize_for_image(np.clip(ceiling - floor, 0, 10), 0, 10)
            Image.fromarray(rgb).save("height.png")
            print(f"Saved height map to height.png")
            print(f"  Ceiling open_sky={(ceiling > 1000).sum()}, Floor open_sky={(floor < -1000).sum()}")
            break

    for name in ["mm.bin", "meta_min.bin"]:
        p = os.path.join(sys.argv[1], name)
        if os.path.exists(p):
            normals = np.fromfile(p, dtype=np.float32).reshape(512, 512, 4)
            rgb = ((normals[:, :, :3] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(rgb).save("meta_min.png")
            print(f"Saved meta_min normals to meta_min.png")
            break

    for name in ["mf.bin", "meta_max.bin"]:
        p = os.path.join(sys.argv[1], name)
        if os.path.exists(p):
            normals = np.fromfile(p, dtype=np.float32).reshape(512, 512, 4)
            Image.fromarray(((normals[:, :, :3] + 1) * 127.5).clip(0, 255).astype(np.uint8)).save("meta_max.png")
            print(f"Saved {name} normals to meta_max.png")
            break

def normalize_for_image(arr, vmin=None, vmax=None):
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return (normalized * 255).astype(np.uint8)

if __name__ == "__main__":
    import sys
    import os
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: read_dumps.py <dump_dir> [output_dir]")
        print("  Dumps: w.bin/winding.bin, h.bin/height.bin, mm.bin/meta_min.bin, mf.bin/meta_max.bin")
        print("  Outputs: PNG files in current directory")
        sys.exit(1)

    main()