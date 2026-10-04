//! `S_RC_SCREENSHOT=path.png cargo run`: saves the window after Sponza has
//! loaded and the cascades have run, then exits, with the heightmap overlay
//! hidden. `S_RC_INDIRECT=0` turns the indirect light off for the shot,
//! `S_RC_PROBES=1` shows the probes.

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};

/// Frames to wait for loading and the heightmap bake.
const SHOT_FRAME: u32 = 200;
const EXIT_FRAME: u32 = 240;

pub struct ScreenshotPlugin;

impl Plugin for ScreenshotPlugin {
    fn build(&self, app: &mut App) {
        if std::env::var_os("S_RC_SCREENSHOT").is_some() {
            app.add_systems(Update, take_screenshot);
        }
    }
}

fn take_screenshot(
    mut commands: Commands,
    mut frames: Local<u32>,
    mut lighting: ResMut<crate::ui::LightingState>,
    mut overlay: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    if *frames == 2 {
        lighting.indirect_enabled = std::env::var("S_RC_INDIRECT").as_deref() != Ok("0");
        lighting.probes_enabled = std::env::var("S_RC_PROBES").as_deref() == Ok("1");
        for mut visibility in &mut overlay {
            *visibility = Visibility::Hidden;
        }
    }
    if *frames == SHOT_FRAME {
        let path = std::env::var("S_RC_SCREENSHOT").unwrap();
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    }
    if *frames == EXIT_FRAME {
        exit.write(AppExit::Success);
    }
}
