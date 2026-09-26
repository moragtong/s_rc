This walkthrough covers the entire 3D real-time Global Illumination engine in `s_rc`, file by file and bit by bit, disregarding `rc2d`.

---

### High-Level Architecture Overview

The codebase implements a **real-time 3D Global Illumination (GI) pipeline** in Bevy 0.16 / WGPU / WGSL based on **Signed Distance Field (SDF) Geometry Clipmaps** and **3D Radiance Cascades**:
1. **Scene & Meshes**: Analytic geometry (test room with doorway and directional sun) represented both as standard raster meshes and CPU-baked mesh distance fields.
2. **Toroidal Geometry Clipmap**: A camera-centered 3D texture ($384^3$ voxels at $12.5\text{ cm}$) storing signed distances, albedo, unshadowed direct surface color, and sun visibility. It updates incrementally via slice-triggered recomposition when the camera crosses voxel boundaries.
3. **Hi-Z Depth Pyramid**: Conservative nearest-depth pyramid pre-pass (max-reduction under reverse-Z) for accelerating screen-space ray tracing.
4. **3D Radiance Cascades**: 4 hierarchical cascade tiers in world space with toroidal probe windows, doubling spacing and quadrupling directional resolution per tier. Ray hits are cached across frames and merged top-down. Cascade 0 is projected into L2 Spherical Harmonics ($9$ RGB coefficients) for fast surface evaluation.
5. **SDF View & Shading**: Direct sphere-tracing of the SDF clipmap with trilinear probe filtering, visibility rejection, and indirect radiance evaluation.

---

### 1. [src/main.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/main.rs) — Engine Initialization & Wiring

#### Purpose
The entry point initializes the Bevy engine, configures the rendering method, registers all custom plugins, and spawns the camera and test scene.

#### Key Mechanics:
- **`DefaultOpaqueRendererMethod::deferred()`**: Forces Bevy's deferred rendering path. This is required because screen-space GI gathers need G-buffer normals and albedo, not merely already-lit colors.
- **Plugins**: Registers the custom subsystem plugins:
  - [`HiZPlugin`](file:///home/michael/Downloads/CG_lectures/s_rc/src/hi_z.rs#L21) & [`HiZDebugPlugin`](file:///home/michael/Downloads/CG_lectures/s_rc/src/hi_z_debug.rs#L22): Hierarchical Z-pyramid generation and debug view.
  - [`ClipmapPlugin`](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs#L44): Toroidal SDF clipmap management.
  - [`SdfViewPlugin`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_view.rs#L28): Interactive raymarching view.
  - [`Cascades3dPlugin`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/mod.rs#L34): 3D Radiance Cascades pipeline.
  - `RenderDiagnosticsPlugin`: Captures GPU timestamp queries across passes.
- **`setup_scene`**: Instantiates `Scene3d::room(0.2)` ($20\text{ cm}$ thick walls), spawns raster mesh entities tagged with `SceneBoxMesh`, inserts the [`SunLight`](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs#L55-L62) resource ($k = 1 / \tan(\theta_{\text{sun}})$ for soft shadow penumbras), and spawns the 3D camera outside the doorway with:
  - `Msaa::Off`: MSAA cannot be used with G-buffers/deferred depth.
  - `DepthPrepass` & `DeferredPrepass`: Populates depth and G-buffer textures.
  - `HiZ`: Triggers Hi-Z pyramid generation for this view.
  - `ClipmapCamera`: Marks this camera as the anchor the SDF clipmap centers on.
- **`log_adapter_info`**: Inspects WGPU adapter capabilities, device type, and backend (e.g. Vulkan / DX12).

---

### 2. [src/scene3d.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs) — Analytic Test Scene & Serialization

#### Purpose
Defines the single source of truth for the 3D test room. The exact same box primitives are used to generate Bevy raster meshes, CPU-baked SDF grids, and Mitsuba 3 JSON scene exports for offline reference validation.

#### Key Types & Functions:
- **[`SceneBox`](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs#L20-L33)**: Axis-aligned bounding box with `center`, `half_size`, and linear RGB `albedo`.
  - `sdf(p)`: Exact analytic signed distance function for an AABB:
    $$\mathbf{q} = |\mathbf{p} - \mathbf{c}| - \mathbf{h},\quad \text{dist} = \|\max(\mathbf{q}, \mathbf{0})\| + \min(\max(q_x, q_y, q_z), 0)$$
- **[`Sun`](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs#L36-L44)**: Directional light definition with unit vector `toward`, linear `irradiance`, and `angular_radius` ($0.27^\circ$, matching the physical sun).
- **[`Scene3d::room(wall_thickness)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs#L51-L93)**: Constructs a closed room ($x, z \in [-4, 4]\text{ m}$, $y \in [0, 3]\text{ m}$) with a ground slab ($24\times24\text{ m}$), ceiling, and walls. The $-X$ wall is split around an open doorway ($1.2\text{ m}$ wide, $2.2\text{ m}$ tall). The sun shines in through the doorway onto the floor, meaning interior walls are illuminated strictly through indirect light.
- **[`Scene3d::export_json`](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs#L109-L138)**: Serializes camera matrices, clipmap window dimensions, and box definitions to JSON for `tools/mitsuba_depth.py` and `tools/mitsuba_compare.py`.
- **[`spawn`](file:///home/michael/Downloads/CG_lectures/s_rc/src/scene3d.rs#L145-L167)**: Spawns Bevy cuboid meshes with `StandardMaterial` tagged with `SceneBoxMesh`, plus a Bevy `DirectionalLight` matching the sun.

---

### 3. [src/sdf_bake.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs) — CPU Mesh Distance Field Baking

#### Purpose
Implements per-mesh Signed Distance Field (SDF) generation on the CPU using `parry3d` (similar to Unreal Engine 5 Lumen's Mesh Distance Fields). Grids are computed in mesh-local space and cached to disk.

#### Key Types & Functions:
- **[`SdfGrid`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs#L18-L32)**: Uniform voxel grid storing `min` (world-space local origin), `voxel` size, `dims` ($XYZ$ count), and flat `values: Vec<f32>`.
  - `cell_center(cell)`: Computes $\mathbf{p} = \text{min} + (\text{cell} + 0.5) \times \text{voxel}$.
- **[`mesh_triangles(mesh)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs#L35-L45)**: Extracts raw vertex positions (`Float32x3`) and $u32$ triangle indices from Bevy's vertex buffers.
- **[`bake(positions, indices, voxel, padding)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs#L48-L100)**:
  1. Builds a `parry3d::shape::TriMesh` with `TriMeshFlags::ORIENTED | TriMeshFlags::MERGE_DUPLICATE_VERTICES`. Merging vertices reconstructs manifold mesh topology from Bevy's split-normal faces, allowing pseudo-normals to reliably evaluate signs.
  2. Expands the bounding box by `padding` voxels (defaults to 3 voxels).
  3. Divides the grid's $Z$-axis slabs across CPU worker threads via `std::thread::scope`.
  4. For each voxel center $\mathbf{p}$, calls `mesh.project_local_point(p, false)` to find nearest surface point $\mathbf{q}$. If `projection.is_inside` is true, stores $-\|\mathbf{p} - \mathbf{q}\|$, else $+\|\mathbf{p} - \mathbf{q}\|$.
- **[`bake_cached`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs#L105-L125)**, **`write_grid`**, **`read_grid`**:
  - Computes a 64-bit hash from the vertices, indices, voxel size, and padding.
  - Stores binary files under `target/sdf_cache/{hash}.sdf` with magic header `SDF1` (32-byte header + $4 \times N$ float bytes), skipping re-computation on subsequent runs.

---

### 4. [src/clipmap.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs) — Toroidal Geometry & Shadow Clipmap

#### Purpose
Manages the primary SDF clipmap: a camera-centered 3D grid of $384^3$ voxels at $12.5\text{ cm}$ resolution (covering a $48\text{ m}$ span). World voxels are mapped toroidally into a 3D texture using `Repeat` addressing. When the camera moves across voxel boundaries, only newly revealed slabs are recomposed on the GPU.

#### Key Structures & Textures:
- **Resolution Constants**:
  - `RESOLUTION = 384`, `VOXEL = 0.125`, `BAKE_VOXEL = 0.0625` (mesh bakes are twice as fine: $6.25\text{ cm}$).
- **[`ClipmapTextures`](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs#L210-L218)**:
  - `sdf` (`R16Float`): Composed signed distance values.
  - `albedo` (`Rgba8Unorm`): Surface albedo sampled from nearest mesh material.
  - `color` (`Rgba16Float`): Unshadowed surface radiance:
    $$\mathbf{L}_{\text{unshadowed}} = \frac{\text{albedo}}{\pi} \cdot \mathbf{E}_{\text{sun}} \cdot \max(\mathbf{n} \cdot \mathbf{\omega}_{\text{sun}}, 0)$$
  - `shadow` (`R16Float`): Directional sun visibility factor $\in [0, 1]$.
  - `sampler`: 3D linear filtering sampler with `AddressMode::Repeat`.
- **[`GpuInstance`](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs#L77-L88)**: Per-mesh instance descriptor uploaded to GPU: `world_to_local` matrix, `grid_min`, `voxel`, `dims`, `offset` in the shared distance buffer, `albedo`, and local `mesh_min`/`mesh_max`.

#### Core Algorithms:
- **`revealed_regions(old, new, res)`**: When the camera moves from `old` voxel origin to `new`, it computes the delta $\Delta = \text{new} - \text{old}$. If $\Delta$ is non-zero, it constructs up to 3 axis-aligned box slabs covering the newly exposed boundary slices, avoiding the need to recompose the full $384^3$ volume.
- **`bake_scene`**: Queries `SceneBoxMesh`, loads or bakes the mesh SDFs via `sdf_bake`, and creates the `SdfScene` resource.
- **`compose_clipmap`**: Dispatches two compute passes on the pending regions:
  1. `clipmap_compose`: Samples per-mesh SDFs, evaluates minimum distance, normals, albedo, and unshadowed direct color.
  2. `clipmap_shadow`: Runs Aaltonen's soft shadow cone-march over the newly composed SDF to compute sun visibility.

---

### 5. [src/hi_z.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/hi_z.rs) — Nearest-Depth Hi-Z Pyramid

#### Purpose
Generates a hierarchical Z-pyramid (downsampling by factors of 2 to $1\times1$) from Bevy's camera prepass depth buffer.

#### Key Mechanics:
- **Reverse-Z Handling**: Modern Bevy uses reverse-Z perspective projections ($1.0 = \text{near}$, $0.0 = \text{far}$). For ray marching, a cell can only be safely skipped if the ray passes entirely in front of everything in that cell. Therefore, the "nearest" depth is computed using a **`max` reduction**, in contrast to occlusion culling pyramids which use `min`.
- **Pipeline Structure**:
  - `first_mip_layout` & `first_mip`: Reads `texture_depth_2d` and writes mip 0 of `R32Float` `hi_z_pyramid`.
  - `downsample_layout` & `downsample`: Successively samples mip $i - 1$ as `texture_2d<f32>` and writes mip $i$.
- **`build_hi_z`**: Dispatches the downsample passes in `Core3dSystems::Prepass` immediately after `late_deferred_prepass`.

---

### 6. [src/hi_z_debug.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/hi_z_debug.rs) — Hi-Z Pyramid Visualization

#### Purpose
A post-processing debug pass that renders a selected Hi-Z mip level full-screen to verify pyramid correctness.

#### Controls & Shading:
- **Keys**: `H` toggles debug display on/off; `[` and `]` step down/up through the mip levels.
- **Pipeline**: Uses Bevy's `FullscreenShader` pipeline, specialized per target swapchain format, running after tonemapping.
- **Color Coding**:
  - Grayscale gradient: Linear camera distance ($0$ to $20\text{ m}$).
  - Red tint: Identifies pixels that are sky ($0.0$) at mip 0 but contain geometry in coarse mips. This confirms that coarser levels conservatively dilate/fatten silhouettes outwards.

---

### 7. [src/cascades3d/layout.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs) — Radiance Cascades Mathematical Layout

#### Purpose
Defines the mathematical parameters and CPU buffers for 3D Radiance Cascades. This must mirror the GPU layout in `cascades3d.wgsl`.

#### Core Constants & Hierarchical Scaling:
- `CASCADES = 4`: Four cascade levels.
- **Cascade 0**:
  - `SPACING0 = 0.25`: Probes every $25\text{ cm}$.
  - `OCTA_SIDE0 = 8`: $8 \times 8 = 64$ directions per probe.
  - `INTERVAL0 = 1.5`: Ray interval $[0.0, 1.5\text{ m}]$.
  - `RADIUS0 = 10.5`: Active probe radius around camera is $10.5\text{ m}$.
- **Scaling Rule per Cascade $n$**:
  - Probe spacing doubles: $s_n = s_0 \cdot 2^n$ ($0.25\text{ m}, 0.5\text{ m}, 1.0\text{ m}, 2.0\text{ m}$).
  - Octahedral side doubles: $\text{side}_n = \text{side}_0 \cdot 2^n$ (yielding $4\times$ directions: $64, 256, 1024, 4096$ rays).
  - Ray distance interval doubles: $[1.5 \cdot 2^{n-1}, 1.5 \cdot 2^n]\text{ m}$.
- **3D Bilinear Fix**: In standard 2D cascades, 1 ray connects to 4 bilinear children. In 3D, each parent ray branches into **8 segments**—one to each of the 8 trilinear child probe centers of the next cascade level. The top cascade has no children, so it traces 1 segment.

#### Octahedral Geometry & Spherical Angles:
- **[`oct_decode(uv)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L106-L115)**: Decodes a $[0, 1]^2$ parameter into a normalized 3D direction vector on the unit sphere.
- **`octa_solid_angles(side)`**: Because octahedral cells are not equal-area on the sphere, this integrates the solid angle of each cell using the Van Oosterom & Strackee spherical triangle area formula:
  $$\tan\left(\frac{E}{2}\right) = \frac{|\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})|}{1 + \mathbf{a} \cdot \mathbf{b} + \mathbf{b} \cdot \mathbf{c} + \mathbf{c} \cdot \mathbf{a}}$$
- **`shading_subsamples`**: Divides each cascade 0 directional cell into $4 \times 4 = 16$ sub-directions with pre-integrated solid angles for cosine-weighted integration during surface evaluation.

---

### 8. [src/cascades3d/mod.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/mod.rs) — 3D Radiance Cascades Compute Orchestrator

#### Purpose
Orchestrates the multi-pass GPU Radiance Cascades pipeline, managing probe selection, ray tracing, caching, top-down radiance gathering, and spherical harmonic projection.

#### Execution Pipeline (Compute Passes):
1. **`select_probes`**: Scans the toroidal window of probes. Any probe within `radius` whose cached tag does not match its current world coordinates is appended to `work_list`.
2. **`write_args`**: Inspects the atomic counter in `work_count` and populates the `dispatch_args` buffer for indirect dispatch.
3. **`trace`**: Indirect compute pass. Marches SDF rays for probes in the `work_list` (8 segments per ray) and packs hit fractions as `unorm16` into the `hits` buffer.
4. **`commit`**: Writes the new world coordinate tags into `tags` so unchanged probes will not be re-traced next frame.
5. **Top-Down Gather Loop (Cascade 3 down to 0)**:
   - **`mark_dirty`**: Marks probes whose cached hits changed or whose child probes changed, building the `gather_list`.
   - **`write_gather_args`**: Formats indirect arguments for the gather pass.
   - **`gather`**: Samples radiance at cached hit positions from the clipmap, reads child cascade radiance, and merges near and far intervals.
   - **`project_sh`** (on cascade 0 only): Projects gathered radiance into 9 L2 Spherical Harmonic coefficients, convolved with clamped cosine factors (Ramamoorthi & Hanrahan), writing into `probe_sh`.

---

### 9. [src/sdf_view.rs](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_view.rs) — SDF Viewport Tracing & Indirect Shading

#### Purpose
Sphere-traces the SDF clipmap directly from the camera at $800 \times 450$ resolution, computes indirect lighting by interpolating cascade 0 probes, and presents the image in a full-window UI node.

#### Key Mechanics:
- **Controls**:
  - `V`: Toggles the SDF view on/off (hides raster meshes and directional shadow maps to eliminate redundant GPU work).
  - `N`: Cycles through 9 debug display modes:
    - `0`: Normals ($\mathbf{n} \cdot 0.5 + 0.5$).
    - `1`: Hit distance.
    - `2`: Sphere-march step count.
    - `3`: Full direct sun + indirect radiance cascades.
    - `4`: Indirect radiance only.
    - `5`: Albedo.
    - `6`: Direct sun irradiance.
    - `7`: Shadow grid texture.
    - `8`: Unshadowed color grid texture.
  - `X`: Exports scene geometry, camera parameters, traced depth buffer, and radiance buffer for Python verification.
- **Trace Dispatch**: `trace_sdf_view` binds the camera uniform, clipmap textures, sampler, and cascade probe SH buffers, dispatching $100 \times 57$ workgroups of $8 \times 8$ threads.

---

### 10. [assets/shaders/clipmap_sample.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/clipmap_sample.wgsl) — Shared Ray Marching & Shadow Functions

#### Purpose
WGSL library imported by all shaders that trace the clipmap (`clipmap_shadow`, `cascades3d`, `sdf_view`).

#### Key Functions:
- **`HIT_FRACTION = 0.25`**: Hit threshold is $\epsilon = 0.25 \times \text{voxel}$. Trilinear filtering rounds convex geometric corners off; a threshold of $0.25 \times \text{voxel}$ prevents grazing rays from skipping sharp box edges.
- **`in_grid`**: Checks bounds with a $1.5\text{-voxel}$ safety margin away from edges to ensure toroidal repeat sampling does not interpolate across opposite window boundaries.
- **`grid_uvw`**: Maps world coordinate $\mathbf{p}$ to toroidal texture coordinate:
  $$\mathbf{uvw} = \frac{\mathbf{p}}{\text{voxel} \cdot \text{resolution}}$$
- **`march_eps`**: Classic sphere-tracer with ray advancement:
  $$t \leftarrow t + \max(d, \text{MIN\_STEP})$$
  Terminates when $d < \epsilon$, $t \ge \text{max\_t}$, or leaving `in_grid`.
- **`sun_visibility`**: Implements **Aaltonen's soft shadow cone estimator** (GDC 2018). Instead of standard point sampling, it triangulates the closest approach between consecutive sphere steps:
  $$y = \min\left(\frac{d^2}{2 \cdot \text{previous}}, d\right),\quad \text{penumbra} = \min\left(\text{res}, \frac{k \sqrt{d^2 - y^2}}{\max(t - y, 10^{-4})}\right)$$
  Followed by a smoothstep curve $x^2(3 - 2x)$.

---

### 11. [assets/shaders/clipmap_compose.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/clipmap_compose.wgsl) — GPU SDF Composition

#### Purpose
Computes signed distance, surface albedo, and unshadowed direct color for each voxel in the pending clipmap region.

#### Key Mechanics:
- **Workgroup**: `@workgroup_size(4, 4, 4)` over the 3D bounding region.
- **`sample_instance(inst, p)`**:
  - Transforms world point $\mathbf{p}$ into mesh-local space.
  - Inside the mesh's baked grid bounds, performs trilinear interpolation of the stored distance field values.
  - Outside the grid bounds, computes exact distance to the mesh's AABB (`length(max(max(min - p, p - max), 0.0))`). This ensures conservative distances that prevent ray overshoot.
- **Surface Band**: For voxels within `SURFACE_BAND = 2.5` voxels of a zero-crossing:
  - Evaluates local surface normal $\mathbf{n}_{\text{local}}$ via central differences of `sample_instance`.
  - Transforms normal to world space using the inverse-transpose:
    $$\mathbf{n}_{\text{world}} = \text{normalize}\left((\mathbf{M}_{\text{world}\to\text{local}}^T \cdot \mathbf{n}_{\text{local}})_{xyz}\right)$$
  - Stores albedo.
  - If $d \ge 0.0$ (in empty space outside the surface, preventing light leakage from exterior sunlit walls into interior rooms), writes Lambertian direct color:
    $$\mathbf{C} = \frac{\text{albedo}}{\pi} \cdot \mathbf{E}_{\text{sun}} \cdot \max(\mathbf{n} \cdot \mathbf{\omega}_{\text{sun}}, 0)$$

---

### 12. [assets/shaders/clipmap_shadow.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/clipmap_shadow.wgsl) — Shadow Grid Generation

#### Purpose
Computes sun visibility for voxels near surfaces in newly composed clipmap regions.

#### Key Mechanics:
- Runs in `@workgroup_size(4, 4, 4)`.
- If $|d| < 2.5 \times \text{voxel}$:
  - Computes gradient normal $\mathbf{n}$ from the composed distance field.
  - Projects back to the true surface: $\mathbf{p}_{\text{surface}} = \mathbf{p} - \mathbf{n} \cdot d$.
  - Offsets $7\text{ cm}$ along the normal to avoid self-shadowing: $\mathbf{p}_{\text{start}} = \mathbf{p}_{\text{surface}} + \mathbf{n} \cdot 0.07$.
  - Calls `sun_visibility()` cone-tracing along the sun vector.
  - Writes the scalar visibility factor into `shadow_out` (`R16Float`).

---

### 13. [assets/shaders/hi_z.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/hi_z.wgsl) — Hi-Z Downsampling Kernel

#### Purpose
Generates conservative mip levels for the hierarchical depth pyramid.

#### Key Mechanics:
- If compiled with `#ifdef FIRST_MIP`, loads full-resolution camera depth from `texture_depth_2d` and writes to mip 0.
- For higher mips, gathers a $2 \times 2$ block from the previous mip level:
  - If the source dimensions are odd, extends the kernel footprint at the screen edge to $3 \times 2$ or $2 \times 3$ to ensure boundary geometry is not discarded.
  - Computes the conservative nearest depth using `max(nearest, sample)` to match reverse-Z conventions.

---

### 14. [assets/shaders/hi_z_debug.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/hi_z_debug.wgsl) — Hi-Z Debug Fragment Shader

#### Purpose
Renders a selected Hi-Z mip to the display for visual inspection.

#### Key Mechanics:
- Linearizes reverse-Z depth:
  $$\text{dist} = \frac{\text{near}}{\max(\text{NDC}_z, 10^{-7})}$$
- Normalizes distance across $[0, \text{max\_distance}]$ to produce a grayscale display (white = close, black = far).
- Compares full-resolution depth with coarse depth: if `full == 0.0` (sky) and `coarse > 0.0` (geometry), blends a red tint. This provides visual verification that silhouettes dilate outwards across coarser mip levels.

---

### 15. [assets/shaders/cascades3d.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/cascades3d.wgsl) — 3D Radiance Cascades Implementation

#### Purpose
The primary compute shader implementing 3D Radiance Cascades over the SDF clipmap.

#### Key Functions & Compute Passes:
- **Direction & Window Math**:
  - `oct_decode`: Reconstructs spherical ray direction from 2D octahedral index.
  - `slot_world` / `world_slot`: Converts between 1D toroidal slot index and 3D world probe grid coordinates.
  - `pack_tag(world)`: Encodes world probe coordinates into a 32-bit tag to track validity.
- **Segment Marching & Hit Caching**:
  - `segment(world, ray, k)`: Computes the 3D line segment for ray segment $k \in [0, 7]$. Segment starts at $\mathbf{p}_{\text{parent}} + \mathbf{d} \cdot r_{\min}$ and ends at $\mathbf{p}_{\text{child}[k]} + \mathbf{d} \cdot r_{\max}$.
  - `trace`: Sphere-marches the segments and packs the hit distance fraction into `hits: array<u32>` using `pack2x16unorm`.
- **Visibility Rejection**:
  - `reachable_children(world)`: Evaluates a 3D visibility mask. A child probe is only accepted if it sits in empty space and has an unobstructed line of sight from the parent probe. This eliminates light leaking across thin walls, floors, and ceilings.
- **Gather & Radiance Merging**:
  - `gather`: Runs top-down. If a cascade has children, it computes `child_arc` (solid-angle-weighted average of child directional cells covering the parent's solid angle) and merges near and far intervals:
    $$\mathbf{L}_{\text{merged}} = \mathbf{L}_{\text{near}} + \alpha_{\text{near}} \cdot \mathbf{L}_{\text{far}},\quad \alpha_{\text{merged}} = \alpha_{\text{near}} \cdot \alpha_{\text{far}}$$
    Near radiance is evaluated at the cached hit point by sampling `color * shadow` half a voxel back from the surface.
- **Spherical Harmonics Projection**:
  - `project_sh`: For cascade 0, projects the gathered directional radiance into 9 L2 Spherical Harmonics ($Y_{lm}$) convolved with clamped cosine lobes (Ramamoorthi & Hanrahan factors: $\pi, \frac{2\pi}{3}, \frac{\pi}{4}$). Packed into 7 `vec4<f32>` values (28 floats) per probe in `probe_sh`.

---

### 16. [assets/shaders/sdf_view.wgsl](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/sdf_view.wgsl) — Real-Time Indirect Shading & Sphere Tracing

#### Purpose
Computes the final screen image by sphere-tracing the SDF clipmap from the camera and shading surface hits using the L2 Spherical Harmonics from cascade 0.

#### Key Mechanics:
- **`on_threshold(hit)`**: Ray marchers stop at varying depths within the hit threshold depending on step size, causing visual rings across flat surfaces. `on_threshold` projects the hit point outward along the surface normal onto the threshold iso-surface:
  $$\mathbf{p} \leftarrow \mathbf{p} + \mathbf{n} \cdot (\epsilon - \text{SDF}(\mathbf{p}))$$
- **`probe_irradiance(slot, n)`**: Evaluates irradiance from probe L2 SH coefficients for surface normal $\mathbf{n}$:
  $$\mathbf{E}(\mathbf{n}) = \sum_{k=0}^{8} \mathbf{c}_k Y_k(\mathbf{n})$$
- **`cascade_irradiance(p, n)`**:
  - Finds the 8 cascade 0 probes surrounding $\mathbf{p} + \mathbf{n} \cdot 0.5 s$.
  - Performs trilinear filtering over valid probes.
  - Rejects probes that are inside geometry, behind the surface plane ($\mathbf{d} \cdot \mathbf{n} \le 0$), or occluded by geometry between the surface and the probe center.
  - Renormalizes weights over remaining valid probes, eliminating light leaks through walls.
- **Lighting Combination**:
  $$\mathbf{L}_{\text{indirect}} = \frac{\text{albedo}}{\pi} \cdot \mathbf{E}_{\text{cascade}},\quad \mathbf{L}_{\text{total}} = \mathbf{L}_{\text{direct\_sun}} + \mathbf{L}_{\text{indirect}}$$
- Displays tonemapped output ($1.0 - e^{-2\mathbf{L}}$) and stores linear radiance and hit depth into GPU readback buffers.

---

### Verification Tools (`tools/`)

The repository includes offline Python scripts that validate the GPU algorithms against ground truth rendered in Mitsuba 3:
- [`tools/mitsuba_depth.py`](file:///home/michael/Downloads/CG_lectures/s_rc/tools/mitsuba_depth.py): Compares the GPU-exported `sdf_depth.f32` buffer against an exact triangle-mesh raycast in Mitsuba, reporting depth discrepancy statistics and missed edges.
- [`tools/mitsuba_compare.py`](file:///home/michael/Downloads/CG_lectures/s_rc/tools/mitsuba_compare.py): Compares the GPU-exported linear radiance `radiance.f32` against a Mitsuba path-traced reference image, computing RMSE and perceptual error metrics.
- [`tools/mitsuba_render.py`](file:///home/michael/Downloads/CG_lectures/s_rc/tools/mitsuba_render.py): Generates reference renders using Mitsuba's path tracer.
- [`tools/sdf_error.py`](file:///home/michael/Downloads/CG_lectures/s_rc/tools/sdf_error.py): Compares baked voxel distances against exact analytic box distances across space.
