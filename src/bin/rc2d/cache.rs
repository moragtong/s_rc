//! CPU side of the probe hit cache: each cascade's probe window (see
//! "probe window" in the shader) and where its tags and cached hits live in
//! their buffers. Must mirror the shader's `sub_count` / `segment_count` /
//! `hit_words` and window mapping.

/// Size of the hit buffer; wgpu's default max storage binding is 128 MiB.
pub const HIT_BUDGET_BYTES: u64 = 128 << 20;
/// Tag words: enough for every cascade's window to span its whole canvas
/// probe grid at s0 = 1 (512^2 + 256^2 + ... < 2 * 512^2).
pub const TAG_CAPACITY_WORDS: u64 = 2 * 512 * 512;

#[derive(Clone, Copy, Debug, Default)]
pub struct CascadeWindow {
    /// Slots per axis.
    pub size: u32,
    /// This cascade's circle radius, px.
    pub radius: f32,
    /// First word of this cascade's tags / cached hits.
    pub tag_base: u32,
    pub hit_base: u32,
}

#[derive(Clone, Debug)]
pub struct CacheLayout {
    pub cascades: Vec<CascadeWindow>,
    pub hit_words: u64,
    /// Cascade-0 radius actually used: the requested one, shrunk if the
    /// cached hits wouldn't fit HIT_BUDGET_BYTES.
    pub radius: f32,
}

pub struct LayoutInputs {
    pub canvas: u32,
    pub probe0_log2: u32,
    pub cascade_count: u32,
    pub per_direction_levels: u32,
    pub bilinear_fix: bool,
    pub radius: f32,
}

impl CacheLayout {
    pub fn new(inputs: &LayoutInputs) -> Self {
        let mut radius = inputs.radius;
        loop {
            let layout = Self::with_radius(inputs, radius);
            if layout.hit_words * 4 <= HIT_BUDGET_BYTES || radius <= 16.0 {
                return layout;
            }
            radius -= 16.0;
        }
    }

    fn with_radius(inputs: &LayoutInputs, radius0: f32) -> Self {
        let mut cascades = Vec::new();
        let (mut radius, mut tag_base, mut hit_base) = (radius0, 0u64, 0u64);
        for n in 0..inputs.cascade_count {
            let spacing = (1u32 << (n + inputs.probe0_log2)) as f32;
            if n > 0 {
                // Room for every valid parent's 4 bilinear children.
                radius += 1.5 * spacing;
            }
            let grid = inputs.canvas >> (n + inputs.probe0_log2);
            // Even, and at least radius/spacing + 2 either side of the
            // window's center probe, so it holds every probe in the circle.
            let size = (2 * ((radius / spacing).ceil() as u32 + 2)).min(grid);
            cascades.push(CascadeWindow {
                size,
                radius,
                tag_base: tag_base as u32,
                hit_base: hit_base as u32,
            });
            let slots = (size * size) as u64;
            tag_base += slots;
            hit_base += slots * (4u64 << (2 * n)) * hit_words(inputs, n) as u64;
        }
        Self {
            cascades,
            hit_words: hit_base,
            radius: radius0,
        }
    }
}

/// u32 words of cached hits per stored ray; mirrors the shader.
fn hit_words(inputs: &LayoutInputs, n: u32) -> u32 {
    let has_children = n + 1 < inputs.cascade_count;
    let subs = if has_children && n < inputs.per_direction_levels { 4 } else { 1 };
    let segments = if has_children && inputs.bilinear_fix { 4 } else { 1 };
    (subs * segments + 1) / 2
}
