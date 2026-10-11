//! Readback from bake cameras: `S_RC_BAKE_VIEW=up|down S_RC_SCREENSHOT=path.png`
//!
//! After the heightmap bake completes, spawns a one-shot orthographic camera
//! matching the bake camera's transform, renders to an image, reads back the
//! GPU texture via Bevy's GPU readback, and saves as PNG.

use bevy::{
    asset::RenderAssetUsages,
    camera::{Camera, ImageRenderTarget, OrthographicProjection, Projection, RenderTarget, ScalingMode},
    camera::visibility::RenderLayers,
    core_pipeline::{
        Core3dSystems,
        deferred::node::late_deferred_prepass,
        prepass::{DeferredPrepass, DepthPrepass},
    },
    image::Image,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, Msaa, RenderVisibleEntities},
    },
};

use crate::heightmap_bake::{BakeState, BakeView};

/// Frames to wait after bake is done before capturing.
const CAPTURE_DELAY: u32 = 5;
const EXIT_FRAME: u32 = 240;

pub struct BakeCameraReadbackPlugin;

impl Plugin for BakeCameraReadbackPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BakeViewConfig::default())
            .add_plugins(ExtractResourcePlugin::<BakeViewConfig>::default())
            .add_systems(Update, setup_bake_view_config)
            .add_systems(Update, capture_bake_camera_view);
    }

    fn finish(&self, app: &mut App) {}
}

/// Configuration extracted from environment variables.
#[derive(Resource, Clone, Default, ExtractResource)]
struct BakeViewConfig {
    active: Option<BakeView>,
    direct_only: bool,
    output_path: Option<String>,
}

fn setup_bake_view_config(
    mut config: ResMut<BakeViewConfig>,
) {
    let bake_view = std::env::var("S_RC_BAKE_VIEW").ok();
    let direct_only = std::env::var("S_RC_DIRECT_ONLY").as_deref() == Ok("1");
    let output_path = std::env::var("S_RC_SCREENSHOT").ok();

    config.active = match bake_view.as_deref() {
        Some("up") => Some(BakeView::Up),
        Some("down") => Some(BakeView::Down),
        _ => None,
    };
    config.direct_only = direct_only;
    config.output_path = output_path;
}

/// Tag component for the one-shot camera to identify it
#[derive(Component)]
struct BakeReadbackCamera {
    config: BakeViewConfig,
}

fn capture_bake_camera_view(
    mut commands: Commands,
    mut triggered: Local<bool>,
    mut frames: Local<u32>,
    bake_state: Res<BakeState>,
    config: Res<BakeViewConfig>,
    bake_cameras: Query<(&GlobalTransform, &BakeView)>,
    camera_query: Query<Entity, With<BakeReadbackCamera>>,
    mut images: ResMut<Assets<Image>>,
) {
    *frames += 1;

    // Only act if we have a valid config and the bake is complete
    if config.active.is_none() || config.output_path.is_none() || !bake_state.is_done() {
        return;
    }

    // Wait a few frames after bake is done for everything to settle
    if *frames < CAPTURE_DELAY {
        return;
    }

    // Already have a capture in progress
    if *triggered || !camera_query.is_empty() {
        return;
    }

    let active = config.active.unwrap();

    // Find the bake camera matching the requested view
    let Some((bake_transform, _)) = bake_cameras.iter().find(|(_, view)| **view == active) else {
        return;
    };

    // Create output image (Rgba8UnormSrgb for PNG)
    let extent = bevy::render::render_resource::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_uninit(
        extent,
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let image_handle = images.add(image);

    // Spawn a one-shot orthographic camera matching the bake camera
    let direction = match active {
        BakeView::Up => 1.0,
        BakeView::Down => -1.0,
    };
    // Create a new transform offset from the bake camera's global transform
    let mut camera_transform = Transform::from_translation(bake_transform.translation());
    camera_transform.rotation = bake_transform.rotation();
    // Offset the camera back from the surface by the near plane distance
    // so it's outside the geometry and can see the lit interior
    camera_transform.translation.y += direction * crate::heightmap::BAKE_NEAR;

    let camera_entity = commands.spawn((
        Camera3d::default(),
        Camera {
            order: 1000, // after all other rendering
            ..default()
        },
        Projection::Orthographic(OrthographicProjection {
            near: crate::heightmap::BAKE_NEAR,
            far: 6.0,
            scaling_mode: ScalingMode::Fixed {
                width: 20.0,
                height: 20.0,
            },
            ..OrthographicProjection::default_2d()
        }),
        Msaa::Off,
        bevy::camera::Hdr,
        DepthPrepass,
        DeferredPrepass,
        camera_transform,
        active, // tag with BakeView so we can identify it
        // Use RenderTarget::Image like the bake cameras do
        RenderTarget::Image(ImageRenderTarget::from(image_handle.clone())),
        // Must include LIGHT_LAYER to see the directional light for direct lighting
        RenderLayers::from_layers(&[crate::heightmap::SCENE_LAYER, crate::heightmap::LIGHT_LAYER]),
    ))
    .observe(on_readback_complete)
    .insert(Readback::texture(image_handle.clone()))
    .insert(BakeReadbackCamera { config: config.clone() })
    .id();

    *triggered = true;
}

fn on_readback_complete(
    trigger: On<ReadbackComplete>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
    camera_query: Query<&BakeReadbackCamera>,
) {
    let Some(camera) = camera_query.get(trigger.entity).ok() else {
        return;
    };
    let Some(output_path) = &camera.config.output_path else {
        return;
    };

    // Clean up
    commands.entity(trigger.entity).despawn();

    // Save as PNG - trigger.data contains the readback data
    let width = 512;
    let height = 512;
    let data = &trigger.data;
    let mut img_data = Vec::with_capacity(data.len());
    for chunk in data.chunks_exact(4) {
        img_data.extend_from_slice(chunk);
    }
    let color_img = image::RgbaImage::from_raw(width, height, img_data);
    if let Some(color_img) = color_img {
        if let Err(e) = color_img.save_with_format(output_path, image::ImageFormat::Png) {
            eprintln!("Failed to save bake camera PNG: {e}");
        } else {
            println!("Bake camera ({:?}) saved to {}", camera.config.active.unwrap(), output_path);
        }
    }

    exit.write(AppExit::Success);
}