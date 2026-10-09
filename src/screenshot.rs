//! `S_RC_SCREENSHOT=path.png cargo run`: saves the window after Sponza has
//! loaded and the cascades have run, then exits, with the heightmap overlay
//! hidden.
//!
//! The `S_RC_*` switches below are read on frame 2 whether or not a shot was
//! asked for, so they also work for plain runs and for timing: they used to
//! live behind the `S_RC_SCREENSHOT` check, which meant `S_RC_INDIRECT=0` and
//! `S_RC_OVERLAY=0` were silently ignored by every run that was not a shot.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

/// Frames to wait for loading and the heightmap bake.
const SHOT_FRAME: u32 = 200;
const EXIT_FRAME: u32 = 240;

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, take_screenshot);
    }
}

fn take_screenshot(
    mut commands: Commands,
    mut frames: Local<u32>,
    mut lighting: ResMut<crate::ui::LightingState>,
    mut cross_layer: ResMut<crate::ui::CrossLayerState>,
    mut blend_toggle: ResMut<crate::ui::BlendToggleState>,
    mut overlay: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    if *frames == 2 {
        lighting.indirect_enabled = std::env::var("S_RC_INDIRECT").as_deref() != Ok("0");
        lighting.probes_enabled = std::env::var("S_RC_PROBES").as_deref() == Ok("1");
        lighting.trilinear_enabled = std::env::var("S_RC_TRILINEAR").as_deref() != Ok("0");
        lighting.bilateral_enabled = std::env::var("S_RC_BILATERAL").as_deref() == Ok("1");
        lighting.normal_bias_enabled = std::env::var("S_RC_NORMAL_BIAS").as_deref() != Ok("0");
        cross_layer.enabled = std::env::var("S_RC_CROSS_LAYER").as_deref() == Ok("1");
        blend_toggle.lowpass = std::env::var("S_RC_LOWPASS").as_deref() != Ok("0");
        lighting.bsp_recon_enabled = std::env::var("S_RC_BSPLINE_RECON").as_deref() == Ok("1");
        // S_RC_OVERLAY=1 keeps the heightmap debug planes visible, which is
        // how you check what the GPU bake actually produced. They are visible
        // by default, so a plain run has to hide them itself.
        if std::env::var("S_RC_OVERLAY").as_deref() != Ok("1") {
            for mut visibility in &mut overlay {
                *visibility = Visibility::Hidden;
            }
        }
    }
    let Ok(path) = std::env::var("S_RC_SCREENSHOT") else {
        return;
    };
    if *frames == SHOT_FRAME {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if *frames == EXIT_FRAME {
        exit.write(AppExit::Success);
    }
}
