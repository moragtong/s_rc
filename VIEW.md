Here are the direct answers to your three questions:

---

### 1. What here can be replaced with an open source library?

Looking across the current custom implementation, there are three primary candidates where mature open-source crates or shaders could replace or streamline custom code:

1. **CPU Mesh Distance Field Baking ([`sdf_bake.rs`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs))**:
   - *Current state*: Custom thread pooling, sign-determination logic, and raw binary caching written on top of `parry3d`.
   - *Alternatives*: 
     - Libraries like [`mesh_to_sdf`](https://crates.io/crates/mesh_to_sdf) or [`fast_sweeping`](https://crates.io/crates/fast_sweeping) specialize in generating sign-accurate dense SDF grids from arbitrary polygon soups and watertight meshes.
     - For serialization/caching, using [`rkyv`](https://crates.io/crates/rkyv) or [`bincode`](https://crates.io/crates/bincode) instead of the manual binary header and raw byte chunking in [`sdf_bake.rs#L127-L162`](file:///home/michael/Downloads/CG_lectures/s_rc/src/sdf_bake.rs#L127-L162).

2. **Spherical Harmonics Math & Projection ([`layout.rs`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs) & [`cascades3d.wgsl`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/cascades3d.wgsl#L330-L388))**:
   - *Current state*: Hand-rolled spherical triangle solid angle integration ([`spherical_triangle_area`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L117-L123)), octahedral mapping, and hardcoded Ramamoorthi-Hanrahan L2 SH coefficients.
   - *Alternatives*:
     - Rust crates like [`spherical_harmonics`](https://crates.io/crates/spherical_harmonics) or [`sh-rs`](https://crates.io/crates/sh-rs) compute analytical coefficients and rotations.
     - Note: The GPU side in WGSL is already very compact (around 60 lines), so replacing the GPU part with an external shader package may add little value compared to keeping it self-contained.

3. **Hi-Z Depth Downsampling ([`hi_z.rs`](file:///home/michael/Downloads/CG_lectures/s_rc/src/hi_z.rs) & [`hi_z.wgsl`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/hi_z.wgsl))**:
   - *Current state*: Custom compute passes generating a depth pyramid.
   - *Why you might keep it custom*: As noted in [`ARCHITECTURE.md`](file:///home/michael/Downloads/CG_lectures/s_rc/ARCHITECTURE.md) and [`hi_z.wgsl`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/hi_z.wgsl#L1-L10), Bevy’s built-in GPU occlusion culling depth pyramid uses a **`min` reduction** (farthest depth in Reverse-Z), whereas raymarching through empty space requires a **conservative nearest `max` reduction**. Most general mipmap generators use linear or box filters, which break geometric depth bounds. However, utilities like FidelityFX SPD (Single Pass Downsampler via WGPU ports) can generate atomic-optimized pyramids in one dispatch.

---

### 2. Low-Res Probe Index Texture (Surface-Only Probes)

Using a low-res probe index texture (or probe indirection table) to allocate probes only near surfaces is an **excellent and standard optimization in modern GI architectures** (such as UE5 Lumen and NVIDIA DDGI). 

#### Why it helps:
In a 3D grid, volume scales cubically ($O(N^3)$), while surfaces scale quadratically ($O(N^2)$):
- Right now, Cascade 0 covers a radius of $10.5\text{ m}$ at $0.25\text{ m}$ spacing ([`RADIUS0`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L22)), creating a toroidal window of around **$88^3 \approx 681,472$ probe positions**.
- In typical scenes (especially indoor rooms like this test scene), **$80\text{–}90\%$ of those probes sit out in completely empty air or deep inside solid geometry**.
- This is explicitly called out in [`cascades3d/mod.rs#L137`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/mod.rs#L137):
  > `hits 1878 MB, radiance 1215 MB` (roughly **$3\text{ GB}$ of VRAM** for inactive/empty space probes!).

#### How to implement it:
1. **Probe Allocation / Classification**:
   - Since you already have the SDF clipmap ([`ClipmapTextures.sdf`](file:///home/michael/Downloads/CG_lectures/s_rc/src/clipmap.rs#L211)), you can evaluate the SDF distance at each probe center in a quick compute pass.
   - Probes with $|\text{SDF}(\mathbf{p})| > \text{threshold}$ (too far into open air or deeply embedded inside walls) are inactive.
2. **Indirection Structure**:
   - A coarse 3D integer texture (e.g. `R32Uint` or `R16Uint` at probe resolution) storing the index into a packed active probe buffer (`probe_index_map(x, y, z) -> active_probe_id`, or `0xFFFFFFFF` for none).
   - Alternatively, an active probe list built via atomic append buffer (`work_list` / `gather_list`, which [`cascades3d.wgsl`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/cascades3d.wgsl#L210) already partially uses).
3. **Trade-offs / Things to Watch Out For**:
   - **Gathering & Merging across Cascades**: In Radiance Cascades, parent probes at Cascade 1 need to query their 8 trilinear child probes at Cascade 0 ([`cascades3d.wgsl#L305-L325`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/cascades3d.wgsl#L305-L325)). If a ray passes through empty space where child probes were skipped, Cascade 1 won't find valid child radiance there unless you either:
     - Keep probes along ray paths (not just surfaces), or
     - Allow the parent cascade to fall back / step further when a child probe is missing (similar to your existing `reachable_children` check in [`reachable_children`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/cascades3d.wgsl#L274-L293)).
   - For final surface evaluation ([`cascade_irradiance`](file:///home/michael/Downloads/CG_lectures/s_rc/assets/shaders/sdf_view.wgsl#L123-L177)), surface hits only query probes within $0.5 \times \text{spacing}$ off the surface, so restricting Cascade 0 probes to a narrow band around surfaces ($|\text{SDF}| \le \text{spacing}$) works directly.

---

### 3. Number of Directions and Density for Each Cascade

The constants and functions controlling direction count, intervals, and probe density are in [**`src/cascades3d/layout.rs`**](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs):

#### Controlling Variables & Functions:
1. **Cascade Count**:
   - [`CASCADES: u32 = 4`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L6)
2. **Probe Density & Window Size**:
   - [`SPACING0: f32 = 0.25`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L12) (Cascade 0 probe spacing in meters; doubles each cascade via [`spacing(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L40)).
   - [`RADIUS0: f32 = 10.5`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L22) (Active probe radius around the camera).
3. **Direction Count**:
   - [`OCTA_SIDE0: u32 = 8`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L15) (The side length of the octahedral directional map for Cascade 0; doubles each cascade via [`octa_side(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L44)).
4. **Distance Intervals**:
   - [`INTERVAL0: f32 = 1.5`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L18) (Calculated via [`interval(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L48)).

#### Exact Breakdown by Cascade Level:

| Cascade ($n$) | Probe Spacing ([`spacing(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L40)) | Density (probes $/ \text{m}^3$) | Octahedral Grid ([`octa_side(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L44)) | Directions per Probe ($\text{side}^2$) | Distance Interval ([`interval(n)`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L48)) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cascade 0** | **$0.25\text{ m}$** | $64\text{ probes/m}^3$ | $8 \times 8$ | **64 directions** | $[0.0\text{ m},\; 1.5\text{ m}]$ |
| **Cascade 1** | **$0.50\text{ m}$** | $8\text{ probes/m}^3$ | $16 \times 16$ | **256 directions** | $[1.5\text{ m},\; 3.0\text{ m}]$ |
| **Cascade 2** | **$1.00\text{ m}$** | $1\text{ probe/m}^3$ | $32 \times 32$ | **1024 directions** | $[3.0\text{ m},\; 6.0\text{ m}]$ |
| **Cascade 3** | **$2.00\text{ m}$** | $0.125\text{ probes/m}^3$ | $64 \times 64$ | **4096 directions** | $[6.0\text{ m},\; 12.0\text{ m}]$ |

*(Note: For Cascade 0 surface shading, each of the 64 direction cells is further sub-divided into $4 \times 4 = 16$ sub-samples via [`SHADE_SUBSAMPLES = 4`](file:///home/michael/Downloads/CG_lectures/s_rc/src/cascades3d/layout.rs#L25) for a total of $1024$ integration directions projected into the 9 L2 Spherical Harmonics).*
