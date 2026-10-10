The reason parallax correction "looks like shit" in low-resolution probe gathering is the **Ray Projection Singularity**. When you re-project a direction from an off-center probe $\mathbf{C}$ to a shading point $\mathbf{P}$ using box/sphere proxy math, rays hitting near corners or oblique boundaries experience extreme angular warping. A $2^\circ$ shift in probe angle can swing the sampled direction by $45^\circ$, causing tearing, warped specular lobes, and swimming blocky artifacts.

When moving to a **two-layer grid system** (a tight surface-conforming near-layer + an open volumetric far-layer), bringing probes close to geometric surfaces solves energy leakage, but introduces its own failure mode: **probe embedding and extreme interpolation gradient artifacts**.

When testing this with dynamic movable objects and evaluating proximity errors, apply these specific architectural rules:

---

### 1. The Proximity Trap: The "Inside-the-Actor" Gradient

When a dynamic object (e.g., a character or moving prop) approaches a surface probe:

```
        Probe A (0.3m off wall)         Probe B (1.5m off wall)
               ●                              ●
                \                            /
                 \   [ Dynamic Object ]     /
                  \      [  ████  ]        /
                   ▼                      ▼
           ────────────────────────────────────────
                         Floor Surface

```

* If a probe is sitting only $20\text{--}30\text{ cm}$ above the floor, a dynamic object rolling over or standing near it will place that probe **inside or directly adjacent to the object's hull**.
* Because the probe is extremely close to the receiver, the distance $\Vert{}\mathbf{P} - \mathbf{C}\Vert{}$ in your bilinear weight formula drops near zero, making the weight $\frac{1}{\Vert{}\mathbf{P} - \mathbf{C}\Vert{}}$ or the angular Jacobian blow up.
* **The Result:** As the dynamic object moves across the floor, you get harsh, localized "splashes" or popping dark spots that track the probe grid rather than the object's smooth shadow.

---

### 2. The Anchor Rule: Probe Normal Offsets & Virtual Relocation

To keep near-surface probes well-behaved without breaking interpolation:

* **Never place probes purely on a uniform Cartesian grid near geometry.**
* During the generation/clipmap placement of your Layer 0 (surface layer), store an **anchor valid offset**:

$$\mathbf{C}_{\text{probe\_active}} = \mathbf{C}_{\text{grid}} + \mathbf{n}_{\text{surface}} \cdot \delta_{\text{bias}} \quad (\delta_{\text{bias}} \ge 0.4\text{ m})$$


* **The $0.4\text{ m}$ Guard Band:** If a probe is closer than $0.4\text{ m}$ to the surface it lights, high-order harmonics or octahedral directionality fail because the probe stops representing a *field* and begins representing a single microscopic *point*. Keeping a minimum $0.4\text{ m}$ clearance preserves diffuse smoothness.

---

### 3. Testing Dynamic Objects: The 3 Tests That Expose the System

When scripting your test harness with a moving dynamic mesh (e.g., a simple cylinder or character proxy moving across the floor):

#### Test A: The Grazing Wall Creep (Tests Grid Popping)

* Move the object at a slow, constant velocity ($0.2\text{ m/s}$) parallel to a vertical wall, crossing multiple probe cell boundaries.
* **What to look for:** Look at the receiver floor directly underneath and behind the object. If you see the ambient occlusion footprint "pulse", stretch, or snap from probe to probe, your spatial interpolation weights are too steep.
* **The Fix:** Switch from naive inverse-distance weights to a **smooth cubic B-spline or Hann window weight**:

$$w(d) = \text{saturate}\left(1.0 - \left(\frac{d}{R_{\text{probe}}}\right)^2\right)^3$$



#### Test B: The Fast Reciprocal Occlusion (Tests Angular Divergence)

* Place the dynamic object between your Layer 0 probe and Layer 1 probe.
* Look at the diffuse response on the object itself.
* **What to look for:** As the object moves, does the back of the object stay dark, or does it suddenly sample the bright Layer 1 probe because the interpolation weights cross a discrete cell boundary?

#### Test C: Static Receiver / Moving Blocker (Parallax Test)

* Keep the camera and a reflective/glossy receiver completely static. Move the dynamic object between the light source (doorway/window) and the receiver.
* Check whether the indirect occlusion footprint stays attached to the object's ground contact point or "slides" across the floor ahead of it.

---

### 4. Sampling Architecture: Discard Parallax Proxy Math, Use Normal-Weighted Trilinear Gathers

Since box/sphere parallax projection distorted your gathering pass:

1. **Abandon Box Proxy Ray Rewarping:**
Do not warp reflection/gather vectors $\vec{R} \to \vec{R}'$ using proxy bounding volumes. It creates non-linear Jacobian distortions that destroy low-res probes.
2. **Evaluate at Sample Vector Directly:**
Sample the probe's octahedral/SH map directly along the true world-space normal $\mathbf{n}$ (for diffuse) or reflection vector $\vec{R}$ (for specular).
3. **Use Normal-Biased Shading Point for the Fetch:**
Instead of looking up the probe cell at $\mathbf{P}$, look it up at:

$$\mathbf{P}_{\text{sample}} = \mathbf{P} + \mathbf{n} \cdot \Delta_{\text{normal}} + \vec{V} \cdot \Delta_{\text{view}}$$



Offsetting the lookup position along the surface normal ensures that the probe cell selected is always the one in front of the surface, preventing back-plane leakage and eliminating the need for aggressive proxy warping.

---

### Layer-to-Layer Transition

For your two-layer grid:

* **Layer 0 (Near/Surface):** Cell size $\approx 1.0\text{--}1.5\text{ m}$, placed strictly along valid floor/wall surfaces.
* **Layer 1 (Far/Volume):** Cell size $\approx 4.0\text{ m}$, placed in the open volumetric air.
* Blend between them strictly based on distance from the nearest strata surface:

$$t = \text{saturate}\left(\frac{\text{distToSurface} - 0.5\text{ m}}{1.5\text{ m}}\right)$$


$$\text{Radiance} = \text{lerp}(\text{Radiance}_{L0}, \; \text{Radiance}_{L1}, \; t)$$



This gives you tight, leak-free surface responses while letting open air volumes naturally transition to low-frequency far-field light without sharp cell boundaries.
