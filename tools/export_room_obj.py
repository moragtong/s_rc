"""Export each material island in `room.glb` as its own OBJ.

Mitsuba's OBJ loader collapses multiple materials in one OBJ into one mesh with
a single diffuse material, so we write one OBJ per obj file and let the scene
dict attach the diffuse material explicitly.
"""
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
GLB = ROOT / "assets/models/room.glb"
OUT_DIR = ROOT / "assets/models/room_obj"


def glb_chunks(path: Path):
    blob = path.read_bytes()
    if blob[:4] != b"glTF":
        raise RuntimeError("not a GLB")
    version, total = struct.unpack_from("<II", blob, 4)
    if version != 2:
        raise RuntimeError(f"GLB version {version}")
    off = 12
    while off < len(blob):
        ln, ty = struct.unpack_from("<II", blob, off)
        yield ln, ty, blob[off + 8 : off + 8 + ln]
        off += 8 + ln


def parse_glb(path: Path):
    blob = path.read_bytes()
    json_chunk = bin_chunk = None
    for ln, ty, data in glb_chunks(path):
        if ty == 0x4E4F534A:
            json_chunk = data.rstrip(b" \t\r\n\x00")
        elif ty == 0x004E4942:
            bin_chunk = data
    if json_chunk is None or bin_chunk is None:
        raise RuntimeError("missing GLB chunk")

    g = json.loads(json_chunk)

    def accessor(a):
        bv = g["bufferViews"][a["bufferView"]]
        off = bv["byteOffset"] + a.get("byteOffset", 0)
        component = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 1, 5126: 3}[
            a["componentType"]
        ]
        dtype = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }[a["componentType"]]
        parts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        x = np.frombuffer(bin_chunk, dtype=dtype, count=a["count"] * parts, offset=off)
        return x.reshape(a["count"], parts)

    return g, accessor


def safe_name(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")


def main():
    g, accessor = parse_glb(GLB)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.obj"):
        old.unlink()
    manifest = {}
    for node_idx in g["scenes"][0]["nodes"]:
        node = g["nodes"][node_idx]
        mesh = g["meshes"][node["mesh"]]
        node_name = node.get("name", f"mesh{node['mesh']}")
        for prim_idx, prim in enumerate(mesh["primitives"]):
            if prim.get("mode", 4) != 4:
                continue
            pos = accessor(g["accessors"][prim["attributes"]["POSITION"]])
            ind = accessor(g["accessors"][prim["indices"]]).reshape(-1)
            material = g["materials"][prim.get("material", 0)]
            albedo = material.get("pbrMetallicRoughness", {}).get(
                "baseColorFactor", [1, 1, 1, 1]
            )

            path = OUT_DIR / f"{safe_name(node_name)}_{prim_idx}.obj"
            with path.open("w") as f:
                f.write(f"o {safe_name(node_name)}_{prim_idx}\n")
                for p in pos:
                    f.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
                normals = []
                for t in range(0, len(ind), 3):
                    tri = ind[t : t + 3]
                    a, b, c = pos[tri[0]], pos[tri[1]], pos[tri[2]]
                    n = np.cross(b - a, c - a)
                    l = np.linalg.norm(n)
                    n = np.array([0.0, 1.0, 0.0], dtype=np.float32) if l == 0 else n / l
                    normals.append(n)
                for n in normals:
                    for _ in range(3):
                        f.write(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n")
                for t in range(len(normals)):
                    tri = ind[3 * t : 3 * t + 3]
                    faces = []
                    for j, v in enumerate(tri):
                        faces.append(f"{int(v) + 1}//{3 * t + j + 1}")
                    f.write("f " + " ".join(faces) + "\n")
            manifest[str(path.relative_to(ROOT).as_posix())] = [
                float(x) for x in albedo[:3]
            ]
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} OBJs under {OUT_DIR}")


if __name__ == "__main__":
    main()
