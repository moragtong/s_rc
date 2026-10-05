//! Room scene: a 280x280 room with 2px diffuse walls and one entrance, a
//! spotlight outside aimed at the wall beside the entrance, and three more
//! lights around the outside. The geometry lives in
//! `assets/shaders/rc2d_room.wgsl`; the constants here must match it.

use bevy::prelude::*;

pub const SCENE_ID: u32 = 1;
pub const SHADER_PATH: &str = "shaders/rc2d_room.wgsl";

/// Spotlight start position (draggable).
pub const LIGHT_START: Vec2 = Vec2::new(130.0, 230.0);

/// The empty inside of the room, walls excluded (wall centers x 180..460,
/// y 140..420, 2px thick).
pub const INTERIOR: Rect = Rect {
    min: Vec2::new(182.0, 142.0),
    max: Vec2::new(458.0, 418.0),
};

/// Error sums for the room's regions, from the per-pixel error readback.
#[derive(Default)]
pub struct RoomStats {
    interior: RegionSums,
    walls: RegionSums,
}

#[derive(Default)]
struct RegionSums {
    squared_error: f64,
    squared_reference: f64,
    reference: f64,
    cascades: f64,
}

impl RegionSums {
    fn add(&mut self, squared_error: f32, squared_reference: f32, reference: f32, cascades: f32) {
        self.squared_error += squared_error as f64;
        self.squared_reference += squared_reference as f64;
        self.reference += reference as f64;
        self.cascades += cascades as f64;
    }

    fn summary(&self, name: &str) -> String {
        format!(
            "{name}: rel. RMSE {:.2}%, cascades {:+.2}% vs reference brightness",
            100.0 * (self.squared_error / self.squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
            100.0 * (self.cascades - self.reference) / self.reference.max(f64::MIN_POSITIVE),
        )
    }
}

impl RoomStats {
    pub fn add_interior(
        &mut self,
        pos: Vec2,
        squared_error: f32,
        squared_reference: f32,
        reference: f32,
        cascades: f32,
    ) {
        if INTERIOR.contains(pos) {
            self.interior
                .add(squared_error, squared_reference, reference, cascades);
        }
    }

    /// Wall pixels show their nearest face's lit color, compared like empty
    /// space. This is the direct check on the bounce lookup: a face that
    /// borrowed the other side's light shows up here first.
    pub fn add_wall(
        &mut self,
        squared_error: f32,
        squared_reference: f32,
        reference: f32,
        cascades: f32,
    ) {
        self.walls
            .add(squared_error, squared_reference, reference, cascades);
    }

    pub fn summary(&self) -> String {
        format!(
            " | {} | {}",
            self.interior.summary("inside room"),
            self.walls.summary("wall faces")
        )
    }
}
