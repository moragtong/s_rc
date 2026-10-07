//! The dual heightmap the 2.5D cascades march: per column, the floor below
//! `SPLIT` and the ceiling above it, each with a normal and an albedo.
//!
//! Filled in by `heightmap_bake`, which rasterises the scene from two
//! orthographic cameras sharing the `SPLIT` plane.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::*,
    },
};

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct Heightmap {
    pub min_bounds: Vec2,
    pub max_bounds: Vec2,
    pub resolution: UVec2,
    pub texture: Handle<Image>,
    pub metadata_min: Handle<Image>,
    pub metadata_max: Handle<Image>,
    pub albedo_min: Handle<Image>,
    pub albedo_max: Handle<Image>,
    /// Sunlit outgoing radiance of the ceiling and floor surfaces, written
    /// on the GPU by the cascades' surface light pass.
    pub radiance_min: Handle<Image>,
    pub radiance_max: Handle<Image>,
    /// 2D generalized winding number of the scene cut at `SPLIT`, written on
    /// the CPU by `winding`. The horizontal containment signal the stored
    /// normals cannot carry: a vertical wall is invisible to a bake whose rays
    /// only run up and down, so both sides of a wall record open air and
    /// `is_solid` never fires there. Same resolution and bounds as the
    /// heightmap, so a heightmap texel indexes it directly.
    pub winding: Handle<Image>,
    /// Highest ceiling or floor in the map; above it rays can't hit.
    pub top: f32,
}

/// Ceiling height of columns with nothing above the split (open sky).
pub const NO_CEILING: f32 = 1.0e4;
/// The height the two bake cameras share, looking in opposite directions:
/// below it is floor, above it is ceiling. Mid-height of the room: `room.glb`
/// spans y -3.0..3.0 (interior -2.7..2.7), so the split has to sit inside that
/// or one of the two cameras starts beyond the geometry and records nothing.
pub const SPLIT: f32 = 0.0;
/// Floor height of columns with nothing below the split.
pub const EMPTY_FLOOR: f32 = SPLIT - 10.0;
/// Near plane offset of both bake cameras from `SPLIT`, along the direction
/// they face. Everything within this distance of the plane is clipped away.
pub const BAKE_NEAR: f32 = 0.05;
/// Resolution along each axis. The map need not be square; it covers
/// `MIN_BOUNDS`..`MAX_BOUNDS`, so its texels need not be either.
pub const RESOLUTION: UVec2 = UVec2::new(512, 512);
/// World extent the map covers, and therefore the metres per texel: 512² over
/// 40x32 m is 0.078 m by 0.0625 m.
///
/// Left at Sponza's extent even though `room.glb` is only 20x20 m, because
/// probe spacing is fixed at `SPACING0` = 1.0 m and every metre-defined reach
/// (`escape_texel`'s 0.75 m, `spacing / 2`) is converted through this. Tightening
/// the bounds to the room shrinks a texel to 0.0195 m, so the same 0.75 m walk
/// spans 38 texels a ring instead of 9.6 — 16x the candidates — and that turned
/// a 76 ms frame into 5 s. The room then fills a twelfth of the map and the rest
/// is open sky, which is what Sponza's 159839 `EMPTY_FLOOR` texels already were.
/// Lower `MIN_CEILING_HITS` instead; see its doc.
pub const MIN_BOUNDS: Vec2 = Vec2::new(-20.0, -16.0);
pub const MAX_BOUNDS: Vec2 = Vec2::new(20.0, 16.0);

/// Render layers, so the heightmap's debug overlay is never baked into the
/// heightmap itself, and so the bake cameras never see a light.
///
/// `SCENE_LAYER` has to be shared with the light: a shadow map renders its
/// casters through the light's own layers, so a light confined to another
/// layer silently shadows nothing. The bake cameras therefore get their own
/// `BAKE_LAYER` instead, which the light is not on — they only need the
/// G-buffer, which is written before lighting.
pub const SCENE_LAYER: usize = 0;
pub const DEBUG_LAYER: usize = 1;
pub const LIGHT_LAYER: usize = 2;
pub const BAKE_LAYER: usize = 3;

pub struct HeightmapPlugin;

impl Plugin for HeightmapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, setup_heightmap);
        app.add_plugins(ExtractResourcePlugin::<Heightmap>::default());
    }
}

fn create_empty_image(resolution: UVec2, format: TextureFormat, channels: usize) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0u8; (resolution.x * resolution.y) as usize * channels * 4],
        format,
        RenderAssetUsages::default(),
    );
    // heightmap_bake writes these with compute; the cascades read them back.
    image.texture_descriptor.usage |=
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image
}

fn setup_heightmap(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let rgba = |images: &mut Assets<Image>| {
        (
            images.add(create_empty_image(
                RESOLUTION,
                TextureFormat::Rgba32Float,
                4,
            )),
            images.add(create_empty_image(
                RESOLUTION,
                TextureFormat::Rgba32Float,
                4,
            )),
        )
    };
    let (metadata_min, metadata_max) = rgba(&mut images);
    let (albedo_min, albedo_max) = rgba(&mut images);
    let (radiance_min, radiance_max) = rgba(&mut images);

    // Read-only on the GPU, and written from the CPU rather than by a
    // compute pass, so it needs no storage binding — R32Float is not a
    // storage format in core WebGPU.
    let mut winding = Image::new(
        Extent3d {
            width: RESOLUTION.x,
            height: RESOLUTION.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0u8; (RESOLUTION.x * RESOLUTION.y) as usize * 4],
        TextureFormat::R32Float,
        RenderAssetUsages::default(),
    );
    winding.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
    let winding = images.add(winding);

    commands.insert_resource(Heightmap {
        min_bounds: MIN_BOUNDS,
        max_bounds: MAX_BOUNDS,
        resolution: RESOLUTION,
        texture: images.add(create_empty_image(RESOLUTION, TextureFormat::Rg32Float, 2)),
        metadata_min,
        metadata_max,
        albedo_min,
        albedo_max,
        radiance_min,
        radiance_max,
        winding,
        top: SPLIT,
    });
}
