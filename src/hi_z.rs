//! Nearest-depth Hi-Z pyramid, rebuilt every frame from the prepass depth.
//!
//! This is the acceleration structure for the on-screen half of the gather:
//! rays march the pyramid, and hits reproject into `SceneColor` / read the
//! G-buffer. See `assets/shaders/hi_z.wgsl` for why this can't reuse Bevy's
//! occlusion-culling depth pyramid (wrong min/max direction).

use bevy::{
    core_pipeline::{
        Core3d, Core3dSystems, deferred::node::late_deferred_prepass, prepass::ViewPrepassTextures,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::{
            binding_types::{texture_2d, texture_depth_2d, texture_storage_2d},
            *,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::{CachedTexture, TextureCache},
    },
};

pub const HI_Z_FORMAT: TextureFormat = TextureFormat::R32Float;
const SHADER_PATH: &str = "shaders/hi_z.wgsl";
const WORKGROUP_SIZE: u32 = 8;

pub struct HiZPlugin;

/// Put on a camera alongside `DepthPrepass` to build a Hi-Z pyramid for it.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct HiZ;

/// Render-world per-view pyramid. Mip 0 is full resolution.
#[derive(Component)]
pub struct ViewHiZ {
    pub texture: CachedTexture,
    pub mip_count: u32,
    /// `[0]` reads the prepass depth; `[i]` reads mip `i - 1`, writes mip `i`.
    bind_groups: Vec<BindGroup>,
}

impl ViewHiZ {
    pub fn mip_size(&self, mip: u32) -> UVec2 {
        let size = self.texture.texture.size();
        UVec2::new((size.width >> mip).max(1), (size.height >> mip).max(1))
    }
}

/// Full chain down to 1x1 for a `size` base level.
pub fn hi_z_mip_count(size: UVec2) -> u32 {
    size.max_element().max(1).ilog2() + 1
}

#[derive(Resource)]
struct HiZPipelines {
    first_mip_layout: BindGroupLayoutDescriptor,
    downsample_layout: BindGroupLayoutDescriptor,
    first_mip: CachedComputePipelineId,
    downsample: CachedComputePipelineId,
}

impl Plugin for HiZPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<HiZ>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_hi_z_pipelines)
            .add_systems(
                Render,
                prepare_hi_z.in_set(RenderSystems::PrepareBindGroups),
            )
            // After the last pass that writes depth, before any lighting
            // that will want to march it.
            .add_systems(
                Core3d,
                build_hi_z
                    .after(late_deferred_prepass)
                    .in_set(Core3dSystems::Prepass),
            );
    }
}

fn init_hi_z_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let dst = texture_storage_2d(HI_Z_FORMAT, StorageTextureAccess::WriteOnly);
    let first_mip_layout = BindGroupLayoutDescriptor::new(
        "hi_z_first_mip_layout",
        &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (texture_depth_2d(), dst)),
    );
    let downsample_layout = BindGroupLayoutDescriptor::new(
        "hi_z_downsample_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                dst,
            ),
        ),
    );

    let shader = asset_server.load(SHADER_PATH);
    let first_mip = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("hi_z_first_mip".into()),
        layout: vec![first_mip_layout.clone()],
        shader: shader.clone(),
        shader_defs: vec!["FIRST_MIP".into()],
        entry_point: Some("downsample".into()),
        ..default()
    });
    let downsample = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("hi_z_downsample".into()),
        layout: vec![downsample_layout.clone()],
        shader,
        entry_point: Some("downsample".into()),
        ..default()
    });

    commands.insert_resource(HiZPipelines {
        first_mip_layout,
        downsample_layout,
        first_mip,
        downsample,
    });
}

fn prepare_hi_z(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<HiZPipelines>,
    views: Query<(Entity, &ViewPrepassTextures), With<HiZ>>,
) {
    for (entity, prepass) in &views {
        let Some(depth) = &prepass.depth else {
            warn_once!("HiZ camera has no DepthPrepass; skipping Hi-Z");
            continue;
        };
        // texture_depth_2d can't bind a multisampled depth buffer. Deferred
        // forces MSAA off anyway, so this only bites forward + MSAA cameras.
        if depth.texture.texture.sample_count() > 1 {
            warn_once!("HiZ requires Msaa::Off; skipping Hi-Z");
            continue;
        }

        let size = Extent3d {
            width: prepass.size.width,
            height: prepass.size.height,
            depth_or_array_layers: 1,
        };
        let mip_count = hi_z_mip_count(UVec2::new(size.width, size.height));

        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("hi_z_pyramid"),
                size,
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: HI_Z_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let mip_views: Vec<TextureView> = (0..mip_count)
            .map(|mip| {
                texture.texture.create_view(&TextureViewDescriptor {
                    label: Some("hi_z_mip"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..default()
                })
            })
            .collect();

        let first_mip_layout = pipeline_cache.get_bind_group_layout(&pipelines.first_mip_layout);
        let downsample_layout = pipeline_cache.get_bind_group_layout(&pipelines.downsample_layout);

        let mut bind_groups = Vec::with_capacity(mip_count as usize);
        bind_groups.push(render_device.create_bind_group(
            "hi_z_first_mip",
            &first_mip_layout,
            &BindGroupEntries::sequential((&depth.texture.default_view, &mip_views[0])),
        ));
        for mip in 1..mip_count as usize {
            bind_groups.push(render_device.create_bind_group(
                "hi_z_downsample",
                &downsample_layout,
                &BindGroupEntries::sequential((&mip_views[mip - 1], &mip_views[mip])),
            ));
        }

        commands.entity(entity).insert(ViewHiZ {
            texture,
            mip_count,
            bind_groups,
        });
    }
}

fn build_hi_z(
    view: ViewQuery<&ViewHiZ>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<HiZPipelines>,
    mut ctx: RenderContext,
) {
    let hi_z = view.into_inner();
    let (Some(first_mip), Some(downsample)) = (
        pipeline_cache.get_compute_pipeline(pipelines.first_mip),
        pipeline_cache.get_compute_pipeline(pipelines.downsample),
    ) else {
        return;
    };

    // GPU timestamp span, for the roadmap's ms-budget profiling.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "hi_z");

    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("hi_z"),
                timestamp_writes: None,
            });

        // One dispatch per mip; wgpu inserts the write->read barrier between
        // dispatches since mip i is storage in one and sampled in the next.
        for (mip, bind_group) in hi_z.bind_groups.iter().enumerate() {
            pass.set_pipeline(if mip == 0 { first_mip } else { downsample });
            pass.set_bind_group(0, bind_group, &[]);
            let size = hi_z.mip_size(mip as u32);
            pass.dispatch_workgroups(
                size.x.div_ceil(WORKGROUP_SIZE),
                size.y.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
    }

    time_span.end(ctx.command_encoder());
}
