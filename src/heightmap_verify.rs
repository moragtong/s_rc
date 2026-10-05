//! Reads the heightmap back from the GPU and decides whether the bake is
//! trustworthy.
//!
//! This is the authority on bake completion, not the frame count. The reduction
//! is idempotent and can land on a frame whose prepasses predate the near-plane
//! caps, and that bake is *not* detectably broken by looking at its size: a
//! capped bake fills 111200 ceiling texels and a stale uncapped one fills
//! 102226, both comfortably over the old threshold. So the check here is not
//! "did it fill enough" but "does it contain the thing we were baking", which
//! the cap texels answer directly.

use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
};

use crate::heightmap::{BAKE_NEAR, CAP_INSET, Heightmap, NO_CEILING, SPLIT};
use crate::heightmap_bake::BakeState;

/// Where caps are written, and how near a ceiling texel must sit to count as
/// one. The 1 cm band is far tighter than the gap between the cap plane and
/// anything else in Sponza, so this cannot fire on scene geometry.
fn cap_plane() -> f32 {
    SPLIT + BAKE_NEAR + CAP_INSET
}

const CAP_TOLERANCE: f32 = 0.01;

/// Cap texels a bake must contain. A capped bake writes about 101000; a stale
/// uncapped one writes 4. Any threshold in that four-order-of-magnitude gap
/// works, and this sits near the bottom so a scene with only a little wall area
/// crossing the split still passes.
const MIN_CAP_HITS: usize = 512;

/// What the readback found, consumed by `verdict`.
#[derive(Resource, Default)]
struct Verdict {
    found: Option<Found>,
}

struct Found {
    hits: usize,
    cap_hits: usize,
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
        false,
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
        false,
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
}

fn report_event(trigger: On<ReadbackComplete>, mut commands: Commands, probes: Query<&Probe>) {
    let Ok(probe) = probes.get(trigger.entity) else {
        return;
    };
    if !probe.report {
        return;
    }
    if probe.tag == "height"
        && probe.chan == 0
        && let Ok(path) = std::env::var("S_RC_DUMP_HEIGHT")
    {
        match std::fs::write(&path, &trigger.data) {
            Ok(()) => info!("wrote heightmap dump ({path})"),
            Err(e) => warn!("heightmap dump failed: {e}"),
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
        let mut cap_hits = 0usize;
        let mut cmin = f32::MAX;
        let mut cmax = f32::MIN;
        let cap = cap_plane();
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
                if (c - cap).abs() <= CAP_TOLERANCE {
                    cap_hits += 1;
                }
            }
        }
        info!(
            "BAKESTATS ceiling open_sky={open} hits_range=[{cmin:.3}, {cmax:.3}] \
             cap_hits={cap_hits}"
        );
        commands.insert_resource(Verdict {
            found: Some(Found {
                hits: res - open,
                cap_hits,
            }),
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
    // The postcondition that actually matters. A bake that ran against prepasses
    // from before the caps arrived is full-sized and passes every size check, so
    // demand the caps themselves: if this scene built cap geometry, the map must
    // show cap texels. Anything less and the reduction read the wrong frame, and
    // a map without caps is worse than no map because it looks finished.
    let caps_present = !bake.caps_expected() || found.cap_hits >= MIN_CAP_HITS;
    if big_enough && caps_present {
        bake.mark_verified();
        info!(
            "heightmap bake verified: {} ceiling hits, {} cap texels",
            found.hits, found.cap_hits
        );
        for entity in &probes {
            commands.entity(entity).despawn();
        }
        return;
    }
    let why = match (big_enough, caps_present) {
        (false, _) => format!("only {} ceiling hits", found.hits),
        (true, false) => format!(
            "{} ceiling hits but only {} cap texels (need {}): the reduction read \
             prepasses from before the caps spawned",
            found.hits, found.cap_hits, MIN_CAP_HITS
        ),
        _ => unreachable!(),
    };
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
