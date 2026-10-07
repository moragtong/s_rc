import trimesh
import numpy as np

# One room, centred on the origin, 20x6x20 m outer with 0.3 m walls, floor and
# ceiling, so the shell is genuinely closed. The inner box must be *shorter*
# than the outer one - making it taller punches through both faces and leaves an
# open tube with nothing for a floor/ceiling bake to record. The `SPLIT` plane
# the bake cameras share sits at y = 0, mid-height, so the room must stay
# centred on the origin.
ROOM = [20.0, 6.0, 20.0]
WALL = 0.3
# Top of the floor slab: ROOM/2 - WALL below the origin.
FLOOR_TOP = -(ROOM[1] / 2.0 - WALL)

# Distinct albedo for each surface, so the six surfaces are identifiable in an
# A/B (parallax vs analytic) by colour alone. These are the material base
# colours the G-buffer albedo reads - the shade pass just multiplies its ground
# truth by them, no normal classification. Later surfaces can carry textures the
# same way; the per-material albedo is the flat-colour special case.
ALBEDO = {
    "floor": [0.85, 0.80, 0.70],
    "ceiling": [0.80, 0.85, 0.90],
    "+x": [0.80, 0.35, 0.30],
    "-x": [0.25, 0.40, 0.75],
    "+z": [0.30, 0.70, 0.30],
    "-z": [0.85, 0.60, 0.20],
}

# A face may be a window or door reveal, so trim its normal by the dominant axis
# and keep the sign: floor (up), ceiling (down), four walls. The count of faces
# per group is diagnostic output, not a constraint.
GROUPS = ["floor", "ceiling", "+x", "-x", "+z", "-z"]

# A few boxes standing on the floor, each its own mesh and its own albedo so the
# indirect A/B can tell them apart. (half-size xyz, centre x, centre z, albedo):
# the centre y is derived from FLOOR_TOP so they sit on the floor, sunk a couple
# of cm into the slab to avoid z-fighting the floor's top face in the bake.
# Every top stays below SPLIT = 0, so the up/down bake reads a box as a raised
# floor in its columns rather than a ceiling.
BOXES = [
    {
        "name": "crate",
        "half": [1.5, 0.6, 1.0],
        "x": -4.5,
        "z": -3.0,
        "albedo": [0.65, 0.25, 0.70],
    },
    {
        "name": "table",
        "half": [0.75, 0.9, 0.75],
        "x": 0.0,
        "z": 0.0,
        "albedo": [0.15, 0.60, 0.65],
    },
    {
        "name": "bench",
        "half": [1.3, 0.45, 0.55],
        "x": 4.2,
        "z": 2.0,
        "albedo": [0.85, 0.30, 0.15],
    },
    {
        "name": "pillar",
        "half": [0.4, 1.1, 0.4],
        "x": 6.8,
        "z": -4.5,
        "albedo": [0.65, 0.75, 0.20],
    },
    {
        "name": "stool",
        "half": [0.5, 0.35, 0.5],
        "x": -1.5,
        "z": 4.2,
        "albedo": [0.55, 0.35, 0.15],
    },
]


def box(extents, center):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


inner = [ROOM[0] - 2 * WALL, ROOM[1] - 2 * WALL, ROOM[2] - 2 * WALL]
room = box(ROOM, [0, 0, 0]).difference(box(inner, [0, 0, 0]))

# Openings, each box extended past the wall it cuts so the boolean does not
# leave a coincident face. The floor top is now at FLOOR_TOP = -2.7, so the
# doorway stands on it: bottom at -2.7, head at -0.3. The windows sit with the
# sill a metre up (y = -1.6) and the head at 0.0, which is a plausible human
# height for the taller room. Walls are 10 m either side of the origin, so the
# long walls take three windows and the side walls one each.
r = ROOM[0] / 2  # 10
sill = -1.6
win_cy = sill + 0.8  # window centre height
openings = [
    ([2.0, 2.4, 1.0], [0.0, FLOOR_TOP + 1.2, r]),  # entrance, +z wall
    ([3.0, 1.6, 1.0], [-6.0, win_cy, -r]),          # window, -z wall
    ([3.0, 1.6, 1.0], [0.0, win_cy, -r]),           # window, -z wall
    ([3.0, 1.6, 1.0], [6.0, win_cy, -r]),           # window, -z wall
    ([1.0, 1.6, 3.0], [r, win_cy, 4.0]),            # window, +x wall
    ([1.0, 1.6, 3.0], [-r, win_cy, -4.0]),          # window, -x wall
]
for extents, center in openings:
    room = room.difference(box(extents, center))

# Split the shell into one mesh per surface orientation. Each group becomes its
# own glTF node with its own material, so the albedo is carried as data
# (material base colours) rather than reconstructed in the shader.
# trim by dominant normal axis, then sign.
groups = {name: [] for name in GROUPS}
fn = room.face_normals
axis = np.abs(fn).argmax(axis=1)
sign = np.sign(fn[np.arange(len(fn)), axis]).astype(int)
for i, a in enumerate(axis):
    if a == 0:
        groups["+x" if sign[i] > 0 else "-x"].append(i)
    elif a == 1:
        groups["floor" if sign[i] > 0 else "ceiling"].append(i)
    else:
        groups["+z" if sign[i] > 0 else "-z"].append(i)

print("face counts per group:")
for name, idx in groups.items():
    print(f"  {name:8s} {len(idx)}")
    idx = np.asarray(idx, dtype=np.int64)

subs = room.submesh([np.asarray(v, dtype=np.int64) for v in groups.values()], append=False)
print("submeshes:", len(subs))

SINK = 0.02
boxes = []
for b in BOXES:
    hx, hy, hz = b["half"]
    m = box([2 * hx, 2 * hy, 2 * hz], [b["x"], FLOOR_TOP + hy - SINK, b["z"]])
    m.name = b["name"]
    boxes.append(m)
    print(f"  box {b['name']:6s} top at y = {FLOOR_TOP + hy - SINK + hy:.2f}")

scene = trimesh.Scene()
for name, m in zip(GROUPS, subs):
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=ALBEDO[name] + [1.0]
        )
    )
    scene.add_geometry(m, node_name=name)
for b in BOXES:
    albedo = b["albedo"]
    mesh = next(x for x in boxes if x.name == b["name"])
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=albedo + [1.0])
    )
    scene.add_geometry(mesh, node_name=b["name"])

scene.export("assets/models/room.glb")
print("wrote assets/models/room.glb")