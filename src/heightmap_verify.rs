//! Reads the heightmap back from the GPU and decides whether the bake is
//! trustworthy.
//!
//! This is the authority on bake completion, not the frame count: the readback
//! is requested only once the reduction is done and the cameras are off, and it
//! is this that releases the rest of the app to treat the map as final.

use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
};

use crate::heightmap::{Heightmap, NO_CEILING};
use crate::heightmap_bake::BakeState;

/// What the readback found, consumed by `verdict`.
#[derive(Resource, Default)]
struct Verdict {
    found: Option<Found>,
}

struct Found {
    hits: usize,
}

/// Set once retries run out, so the warning is a conclusion rather than a
/// per-frame log line for as long as the window stays open.
#[derive(Resource, Default)]
struct GaveUp {
    set: bool,
}

#[derive(Component)]
struct Probe {
    tag: &'static str,
    chan: usize,
    bytes_per_texel: usize,
    report: bool,
}

fn request(
    mut commands: Commands,
    heightmap: Res<Heightmap>,
    bake_done: Res<BakeState>,
    done: Query<(), With<Probe>>,
) {
    // Only once the reduction has run several times: a readback requested any
    // earlier reads the textures as they were initialised, which looks exactly
    // like "the bake wrote nothing".
    if !bake_done.is_done() || !done.is_empty() || bake_done.is_verified() {
        return;
    }
    let spawn = |commands: &mut Commands, tag, chan, bpt, report, handle| {
        commands
            .spawn(Probe {
                tag,
                chan,
                bytes_per_texel: bpt,
                report,
            })
            .observe(report_event)
            .insert(Readback::texture(handle));
    };
    spawn(
        &mut commands,
        "height",
        0,
        8,
        true,
        heightmap.texture.clone(),
    );
    spawn(
        &mut commands,
        "height",
        1,
        8,
        true,
        heightmap.texture.clone(),
    );
    spawn(
        &mut commands,
        "meta_min",
        0,
        16,
        true,
        heightmap.metadata_min.clone(),
    );
    spawn(
        &mut commands,
        "meta_min_Y",
        1,
        16,
        false,
        heightmap.metadata_min.clone(),
    );
    spawn(
        &mut commands,
        "meta_max",
        0,
        16,
        true,
        heightmap.metadata_max.clone(),
    );
    spawn(
        &mut commands,
        "meta_max_Y",
        1,
        16,
        false,
        heightmap.metadata_max.clone(),
    );
    spawn(
        &mut commands,
        "albedo_min",
        0,
        16,
        false,
        heightmap.albedo_min.clone(),
    );
    spawn(
        &mut commands,
        "albedo_max",
        0,
        16,
        false,
        heightmap.albedo_max.clone(),
    );
    spawn(
        &mut commands,
        "radiance_min",
        0,
        16,
        false,
        heightmap.radiance_min.clone(),
    );
    spawn(
        &mut commands,
        "radiance_max",
        0,
        16,
        false,
        heightmap.radiance_max.clone(),
    );
    // The winding map is the one channel written by the CPU rather than by a
    // compute pass, so it is the only one whose readback is checking that a
    // Rust routine ran at all rather than that a shader compiled. Whole-image
    // dump is `S_RC_DUMP_WINDING`.
    spawn(
        &mut commands,
        "winding",
        0,
        4,
        true,
        heightmap.winding.clone(),
    );
}

fn report_event(trigger: On<ReadbackComplete>, mut commands: Commands, probes: Query<&Probe>) {
    let Ok(probe) = probes.get(trigger.entity) else {
        return;
    };
    if !probe.report {
        return;
    }
    // Dump the whole image for whichever tag the caller named, so a channel can
    // be examined off the GPU without a bespoke probe: `S_RC_DUMP_HEIGHT`,
    // `S_RC_DUMP_META_MIN`, `S_RC_DUMP_META_MAX`, `S_RC_DUMP_ALBEDO_MIN`...
    // One write per tag: every tag is read back once per channel, and the
    // channel only selects which float the stats below read.
    if probe.chan == 0
        && let Ok(path) = std::env::var(format!("S_RC_DUMP_{}", probe.tag.to_uppercase()))
    {
        match std::fs::write(&path, &trigger.data) {
            Ok(()) => info!("wrote {} dump ({path})", probe.tag),
            Err(e) => warn!("{} dump failed: {e}", probe.tag),
        }
    }

    let res = 512 * 512;
    let mut mn = f32::MAX;
    let mut mx = f32::MIN;
    let mut nonzero = 0usize;
    let mut nonfinite = 0usize;
    for i in 0..res {
        let base = i * probe.bytes_per_texel + probe.chan * 4;
        let v = f32::from_le_bytes([
            trigger.data[base],
            trigger.data[base + 1],
            trigger.data[base + 2],
            trigger.data[base + 3],
        ]);
        if !v.is_finite() {
            nonfinite += 1;
            continue;
        }
        if v.abs() > 1e-6 {
            nonzero += 1;
        }
        mn = mn.min(v);
        mx = mx.max(v);
    }
    info!(
        "BAKESTATS {} ch{} nonzero={nonzero}/{res} range=[{mn:.4}, {mx:.4}] nonfinite={nonfinite}",
        probe.tag, probe.chan
    );
    if probe.tag == "height" {
        let mut open = 0usize;
        let mut cmin = f32::MAX;
        let mut cmax = f32::MIN;
        for i in 0..res {
            let base = i * 8;
            let c = f32::from_le_bytes([
                trigger.data[base],
                trigger.data[base + 1],
                trigger.data[base + 2],
                trigger.data[base + 3],
            ]);
            if c >= NO_CEILING * 0.5 {
                open += 1;
            } else {
                cmin = cmin.min(c);
                cmax = cmax.max(c);
            }
        }
        info!("BAKESTATS ceiling open_sky={open} hits_range=[{cmin:.3}, {cmax:.3}]");
        commands.insert_resource(Verdict {
            found: Some(Found { hits: res - open }),
        });
    }
}

fn verdict(
    bake: Res<BakeState>,
    mut v: ResMut<Verdict>,
    mut gave_up: ResMut<GaveUp>,
    mut commands: Commands,
    probes: Query<Entity, With<Probe>>,
) {
    let Some(found) = v.found.take() else {
        return;
    };
    if gave_up.set {
        return;
    }
    let big_enough = found.hits as u32 >= BakeState::min_ceiling_hits();
    if big_enough {
        bake.mark_verified();
        info!("heightmap bake verified: {} ceiling hits", found.hits);
        for entity in &probes {
            commands.entity(entity).despawn();
        }
        return;
    }
    let why = format!("only {} ceiling hits", found.hits);
    if !bake.request_retry() {
        warn!(
            "heightmap bake unsound ({why}) after {} attempts; leaving the map \
             as-is",
            bake.attempts()
        );
        gave_up.set = true;
        return;
    }
    info!(
        "heightmap bake attempt {} unsound ({why}); retrying",
        bake.attempts() + 1
    );
    // Drop the probes so `request` can read the map again next attempt.
    for entity in &probes {
        commands.entity(entity).despawn();
    }
}
pub struct HeightmapVerifyPlugin;

impl Plugin for HeightmapVerifyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Verdict>()
            .init_resource::<GaveUp>()
            .add_systems(Update, request)
            .add_systems(Update, verdict.after(request));
    }
}
