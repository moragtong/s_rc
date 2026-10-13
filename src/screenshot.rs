//! `S_RC_SCREENSHOT=path.png cargo run`: saves the window after Sponza has
//! settled and exits, so an A/B visual diff is a one-liner without having
//! to click around in the UI or keep a window open.
//!
//! The `S_RC_*` switches below are read on frame 2 whether or not a shot was
//! requested, so an A/B run without the screenshot tool still gets the flags
//! set up without manual keypresses on startup. They previously lived behind
//! the `S_RC_SCREENSHOT` check, which meant `S_RC_INDIRECT=0` and
//! `S_RC_OVERLAY=0` were silently ignored by every run that was not a shot.
//!
//! `S_RC_DIRECT_ONLY=1` disables indirect lighting (useful for bake camera views).

use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, take_screenshot);
    }
}

fn take_screenshot(
    mut commands: Commands,
    mut frames: Local<u32>,
    windows: Query<Entity, With<Window>>,
    mut lighting: ResMut<crate::ui::LightingState>,
    mut blend_toggle: ResMut<crate::ui::BlendToggleState>,
    mut overlay: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    if *frames == 2 {
        lighting.indirect_enabled = std::env::var("S_RC_INDIRECT").as_deref() != Ok("0");
        lighting.probes_enabled = std::env::var("S_RC_PROBES").as_deref() == Ok("1");
        lighting.normal_bias_enabled = std::env::var("S_RC_NORMAL_BIAS").as_deref() != Ok("0");
        lighting.use_vis_mask = std::env::var("S_RC_VIS_MASK").as_deref() != Ok("0");
        blend_toggle.lowpass = std::env::var("S_RC_LOWPASS").as_deref() != Ok("0");
        if let Ok(passes_str) = std::env::var("S_RC_LOWPASS_PASSES") {
            if let Ok(passes) = passes_str.parse::<u32>() {
                blend_toggle.blur_passes = passes;
                crate::cascades2_5d::LOWPASS_BLUR_PASSES
                    .store(passes, std::sync::atomic::Ordering::Relaxed);
            }
        }
        lighting.bevy_ssao_enabled = std::env::var("S_RC_SSAO").as_deref() != Ok("0");
        // S_RC_DIRECT_ONLY=1 disables indirect lighting
        if std::env::var("S_RC_DIRECT_ONLY").as_deref() == Ok("1") {
            lighting.indirect_enabled = false;
        }
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
    if *frames != 30 {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
    exit.write(AppExit::Success);
}
