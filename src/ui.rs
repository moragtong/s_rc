use std::sync::atomic::Ordering;
use bevy::prelude::*;
use crate::cascades2_5d::{
    BOUNCE_COUNT, CROSS_LAYER_RADIUS_TENTHS, LOWPASS_BLUR_PASSES, MAX_BLUR_PASSES,
    MAX_BOUNCE_COUNT, MAX_CROSS_LAYER_TENTHS,
};

/// Shading pass configuration, extracted to the render app.
#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]
pub struct LightingState {
    /// Indirect lighting enabled.
    pub indirect_enabled: bool,
    /// Probes visualization overlay enabled.
    pub probes_enabled: bool,
    /// Offset the probe-field sample position along the surface normal
    /// (`BIAS_NORMAL`) plus a fraction of the view direction (`BIAS_VIEW`).
    pub normal_bias_enabled: bool,
    /// Use precalculated texture mask for visibility test (`probe_is_visible`),
    /// false means use dynamic per-tap ray-tracing (`probe_visible`).
    pub use_vis_mask: bool,
    /// Bevy standard SSAO pass on/off.
    pub bevy_ssao_enabled: bool,
}

impl Default for LightingState {
    fn default() -> Self {
        Self {
            indirect_enabled: true,
            probes_enabled: false,
            normal_bias_enabled: true,
            use_vis_mask: true,
            bevy_ssao_enabled: true,
        }
    }
}

/// The `blend_sh` toggle, extracted to the render app where the blend
/// dispatch reads it (same shape as the other extracted toggle resources).
/// The height gate and pre-baked line-of-sight visibility check are active inside `blend_sh`.
#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]
pub struct BlendToggleState {
    /// Spatial low-pass pre-filter on the SH coefficients in `blend_sh`.
    /// Off means the plain cross-fade, bit identical to the pre-filter frames.
    pub lowpass: bool,
    /// Number of spatial blur iterations applied.
    pub blur_passes: u32,
    /// Vertical cross-layer blur radius in meters.
    pub cross_layer_radius: f32,
    /// Number of diffuse bounces (1 to 4).
    pub bounces: u32,
    /// Temporal bounce accumulation across relight intervals.
    pub temporal_bounce: bool,
}

impl Default for BlendToggleState {
    fn default() -> Self {
        let temporal_bounce = std::env::var("S_RC_TEMPORAL_BOUNCE")
            .map(|s| s != "0" && s != "false" && s != "FALSE")
            .unwrap_or(true);
        Self {
            lowpass: true,
            blur_passes: 7,
            cross_layer_radius: 2.0,
            bounces: 1,
            temporal_bounce,
        }
    }
}

#[derive(Component)]
pub struct ActionButton(pub Action);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Action {
    ToggleHeightmap,
    ToggleIndirect,
    ToggleProbes,
    ToggleVisMask,
    ToggleNormalBias,
    ToggleLowPass,
    ToggleSsoA,
    ToggleTemporalBounce,
}

#[derive(Component)]
pub struct FpsCounter;

#[derive(Component)]
pub struct BlurPassesLabel;

#[derive(Component)]
pub struct PassesTrack;

#[derive(Component)]
pub struct PassesFill;

#[derive(Component)]
pub struct PassesStepButton(pub i32);

#[derive(Component)]
pub struct CrossLayerLabel;

#[derive(Component)]
pub struct CrossLayerTrack;

#[derive(Component)]
pub struct CrossLayerFill;

#[derive(Component)]
pub struct CrossLayerStepButton(pub i32);

#[derive(Component)]
pub struct BouncesLabel;

#[derive(Component)]
pub struct BouncesTrack;

#[derive(Component)]
pub struct BouncesFill;

#[derive(Component)]
pub struct BouncesStepButton(pub i32);

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightingState>()
            .init_resource::<BlendToggleState>()
            .add_plugins((
                bevy::render::extract_resource::ExtractResourcePlugin::<LightingState>::default(),
                bevy::render::extract_resource::ExtractResourcePlugin::<BlendToggleState>::default(),
            ))
            .add_systems(Startup, setup_ui)
            .add_systems(Update, (button_system, fps_system, slider_system));
    }
}

/// Set false to skip rendering the on-screen toggle panel. The panel is drawn
/// over the scene, so its button pixels land in every screenshot and pollute an
/// A/B diff (`S_RC_INDIRECT`-style comparisons are on the scene, not the UI);
/// with it off the resources are still registered, only the nodes are not
/// spawned. Rebuild to toggle; the panel's own F/G keys still work when it is
/// on.
pub const SHOW_TOGGLE_PANEL: bool = true;

fn setup_ui(mut commands: Commands) {
    if !SHOW_TOGGLE_PANEL {
        return;
    }

    // Top-left corner: Action toggles
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                ..default()
            },
            Pickable::IGNORE,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                    ActionButton(Action::ToggleHeightmap),
                ))
                .with_child((Text::new("Toggle Heightmap (M)"), TextColor(Color::WHITE)));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleIndirect),
                ))
                .with_child((
                    Text::new("Toggle Indirect Lighting (I)"),
                    TextColor(Color::WHITE),
                ));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                    ActionButton(Action::ToggleProbes),
                ))
                .with_child((Text::new("Toggle Probes (P)"), TextColor(Color::WHITE)));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleVisMask),
                ))
                .with_child((
                    Text::new("Probe Visibility Mask (T)"),
                    TextColor(Color::WHITE),
                ));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleNormalBias),
                ))
                .with_child((
                    Text::new("Normal-Biased Fetch (N)"),
                    TextColor(Color::WHITE),
                ));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleLowPass),
                ))
                .with_child((Text::new("Low-Pass Filter (F)"), TextColor(Color::WHITE)));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleSsoA),
                ))
                .with_child((Text::new("Bevy SSAO (U)"), TextColor(Color::WHITE)));
            parent
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::all(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                    ActionButton(Action::ToggleTemporalBounce),
                ))
                .with_child((Text::new("Temporal Bounce (B)"), TextColor(Color::WHITE)));
        });

    // Top-right corner: FPS Counter
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                padding: UiRect::px(10.0, 10.0, 6.0, 6.0),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.65)),
            Pickable::IGNORE,
        ))
        .with_child((
            Text::new("FPS: --"),
            TextColor(Color::WHITE),
            TextFont::from_font_size(15.0),
            FpsCounter,
        ));

    // Bottom-right corner: Controls panel (Blur Passes, Cross-Layer Radius, Bounces)
    let initial_passes = LOWPASS_BLUR_PASSES
        .load(Ordering::Relaxed)
        .clamp(1, MAX_BLUR_PASSES as u32);
    let passes_pct = (initial_passes.saturating_sub(1)) as f32
        / (MAX_BLUR_PASSES - 1) as f32
        * 100.0;

    let initial_radius_tenths = CROSS_LAYER_RADIUS_TENTHS
        .load(Ordering::Relaxed)
        .clamp(0, MAX_CROSS_LAYER_TENTHS);
    let radius_pct = (initial_radius_tenths as f32 / MAX_CROSS_LAYER_TENTHS as f32) * 100.0;
    let radius_m = initial_radius_tenths as f32 / 10.0;

    let initial_bounces = BOUNCE_COUNT
        .load(Ordering::Relaxed)
        .clamp(1, MAX_BOUNCE_COUNT);
    let bounces_pct = (initial_bounces.saturating_sub(1)) as f32
        / (MAX_BOUNCE_COUNT - 1) as f32
        * 100.0;

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                padding: UiRect::all(Val::Px(12.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.08, 0.08, 0.88)),
        ))
        .with_children(|parent| {
            // --- Section 1: Diffuse Bounces ---
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|sec| {
                    sec.spawn((
                        Text::new(format!("GI Bounces: {}", initial_bounces)),
                        TextColor(Color::srgb(1.0, 0.8, 0.4)),
                        TextFont::from_font_size(13.5),
                        BouncesLabel,
                    ));
                    sec.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            BouncesStepButton(-1),
                        ))
                        .with_child((
                            Text::new("-"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(160.0),
                                height: Val::Px(14.0),
                                border_radius: BorderRadius::all(Val::Px(7.0)),
                                overflow: Overflow::clip(),
                                justify_content: JustifyContent::FlexStart,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                            BouncesTrack,
                        ))
                        .with_children(|track| {
                            track.spawn((
                                Node {
                                    width: Val::Percent(bounces_pct),
                                    height: Val::Percent(100.0),
                                    border_radius: BorderRadius::all(Val::Px(7.0)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(1.0, 0.7, 0.2)),
                                BouncesFill,
                                Pickable::IGNORE,
                            ));
                        });

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            BouncesStepButton(1),
                        ))
                        .with_child((
                            Text::new("+"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));
                    });
                });

            // --- Section 2: Blur Passes ---
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|sec| {
                    sec.spawn((
                        Text::new(format!("Blur Passes: {}", initial_passes)),
                        TextColor(Color::WHITE),
                        TextFont::from_font_size(13.5),
                        BlurPassesLabel,
                    ));
                    sec.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            PassesStepButton(-1),
                        ))
                        .with_child((
                            Text::new("-"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(160.0),
                                height: Val::Px(14.0),
                                border_radius: BorderRadius::all(Val::Px(7.0)),
                                overflow: Overflow::clip(),
                                justify_content: JustifyContent::FlexStart,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                            PassesTrack,
                        ))
                        .with_children(|track| {
                            track.spawn((
                                Node {
                                    width: Val::Percent(passes_pct),
                                    height: Val::Percent(100.0),
                                    border_radius: BorderRadius::all(Val::Px(7.0)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.35, 0.75, 0.35)),
                                PassesFill,
                                Pickable::IGNORE,
                            ));
                        });

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            PassesStepButton(1),
                        ))
                        .with_child((
                            Text::new("+"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));
                    });
                });

            // --- Section 3: Cross-Layer Radius ---
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|sec| {
                    sec.spawn((
                        Text::new(if radius_m <= 0.001 {
                            "Cross-Layer: 0.0 m (isolated)".to_string()
                        } else {
                            format!("Cross-Layer Radius: {:.1} m", radius_m)
                        }),
                        TextColor(Color::WHITE),
                        TextFont::from_font_size(13.5),
                        CrossLayerLabel,
                    ));
                    sec.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            CrossLayerStepButton(-1),
                        ))
                        .with_child((
                            Text::new("-"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(160.0),
                                height: Val::Px(14.0),
                                border_radius: BorderRadius::all(Val::Px(7.0)),
                                overflow: Overflow::clip(),
                                justify_content: JustifyContent::FlexStart,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                            CrossLayerTrack,
                        ))
                        .with_children(|track| {
                            track.spawn((
                                Node {
                                    width: Val::Percent(radius_pct),
                                    height: Val::Percent(100.0),
                                    border_radius: BorderRadius::all(Val::Px(7.0)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.35, 0.65, 0.85)),
                                CrossLayerFill,
                                Pickable::IGNORE,
                            ));
                        });

                        row.spawn((
                            Button,
                            Node {
                                width: Val::Px(26.0),
                                height: Val::Px(22.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            CrossLayerStepButton(1),
                        ))
                        .with_child((
                            Text::new("+"),
                            TextColor(Color::WHITE),
                            TextFont::from_font_size(15.0),
                        ));
                    });
                });
        });
}

fn button_system(
    mut interaction_query: Query<(&Interaction, &mut BackgroundColor, &ActionButton), With<Button>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut heightmap_query: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut lighting_state: ResMut<LightingState>,
    mut blend_toggle: ResMut<BlendToggleState>,
) {
    let mut toggle_h = keys.just_pressed(KeyCode::KeyM);
    let mut toggle_i = keys.just_pressed(KeyCode::KeyI);
    let mut toggle_p = keys.just_pressed(KeyCode::KeyP);
    let mut toggle_vis = keys.just_pressed(KeyCode::KeyT);
    let mut toggle_bias = keys.just_pressed(KeyCode::KeyN);
    let mut toggle_lp = keys.just_pressed(KeyCode::KeyF);
    let mut toggle_ssao = keys.just_pressed(KeyCode::KeyU);
    let mut toggle_tb = keys.just_pressed(KeyCode::KeyB);

    for (interaction, mut color, action) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            match action.0 {
                Action::ToggleHeightmap => toggle_h = true,
                Action::ToggleIndirect => toggle_i = true,
                Action::ToggleProbes => toggle_p = true,
                Action::ToggleVisMask => toggle_vis = true,
                Action::ToggleNormalBias => toggle_bias = true,
                Action::ToggleLowPass => toggle_lp = true,
                Action::ToggleSsoA => toggle_ssao = true,
                Action::ToggleTemporalBounce => toggle_tb = true,
            }
        }
        match action.0 {
            Action::ToggleIndirect => {
                color.0 = if lighting_state.indirect_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleProbes => {
                color.0 = if lighting_state.probes_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleVisMask => {
                color.0 = if lighting_state.use_vis_mask {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleNormalBias => {
                color.0 = if lighting_state.normal_bias_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleLowPass => {
                color.0 = if blend_toggle.lowpass {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleSsoA => {
                color.0 = if lighting_state.bevy_ssao_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            Action::ToggleTemporalBounce => {
                color.0 = if blend_toggle.temporal_bounce {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                };
            }
            _ => {}
        }
    }

    if toggle_h {
        if let Ok(mut vis) = heightmap_query.single_mut() {
            *vis = match *vis {
                Visibility::Hidden => Visibility::Visible,
                _ => Visibility::Hidden,
            };
        }
    }

    if toggle_i {
        lighting_state.indirect_enabled = !lighting_state.indirect_enabled;
        bevy::log::info!("Indirect diffuse: {}", lighting_state.indirect_enabled);
    }

    if toggle_p {
        lighting_state.probes_enabled = !lighting_state.probes_enabled;
        bevy::log::info!("Probes overlay: {}", lighting_state.probes_enabled);
    }

    if toggle_vis {
        lighting_state.use_vis_mask = !lighting_state.use_vis_mask;
        bevy::log::info!(
            "Probe visibility method: {}",
            if lighting_state.use_vis_mask {
                "baked mask (probe_is_visible)"
            } else {
                "dynamic ray march (probe_visible)"
            }
        );
    }

    if toggle_bias {
        lighting_state.normal_bias_enabled = !lighting_state.normal_bias_enabled;
        bevy::log::info!(
            "Normal-biased fetch: {}",
            lighting_state.normal_bias_enabled
        );
    }

    if toggle_lp {
        blend_toggle.lowpass = !blend_toggle.lowpass;
        bevy::log::info!("Low-pass spatial filter: {}", blend_toggle.lowpass);
    }

    if toggle_ssao {
        lighting_state.bevy_ssao_enabled = !lighting_state.bevy_ssao_enabled;
        bevy::log::info!("Bevy SSAO: {}", lighting_state.bevy_ssao_enabled);
    }

    if toggle_tb {
        blend_toggle.temporal_bounce = !blend_toggle.temporal_bounce;
        bevy::log::info!("Temporal bounce: {}", blend_toggle.temporal_bounce);
    }
}

#[derive(Default)]
struct FpsTracker {
    frames: u32,
    accum_time: f32,
}

fn fps_system(
    time: Res<Time>,
    mut query: Query<&mut Text, With<FpsCounter>>,
    mut tracker: Local<FpsTracker>,
) {
    tracker.frames += 1;
    tracker.accum_time += time.delta_secs();
    if tracker.accum_time >= 0.25 {
        let fps = tracker.frames as f32 / tracker.accum_time.max(0.0001);
        let ms = (tracker.accum_time / tracker.frames as f32) * 1000.0;
        for mut text in &mut query {
            text.0 = format!("{:.0} FPS ({:.1} ms)", fps, ms);
        }
        tracker.frames = 0;
        tracker.accum_time = 0.0;
    }
}

#[derive(bevy::ecs::system::SystemParam)]
struct SliderQueries<'w, 's> {
    passes_buttons: Query<'w, 's, (&'static Interaction, &'static PassesStepButton), (With<Button>, Changed<Interaction>)>,
    layer_buttons: Query<'w, 's, (&'static Interaction, &'static CrossLayerStepButton), (With<Button>, Changed<Interaction>)>,
    bounce_buttons: Query<'w, 's, (&'static Interaction, &'static BouncesStepButton), (With<Button>, Changed<Interaction>)>,
    passes_track: Query<'w, 's, (&'static Interaction, &'static ComputedNode, &'static GlobalTransform), With<PassesTrack>>,
    layer_track: Query<'w, 's, (&'static Interaction, &'static ComputedNode, &'static GlobalTransform), With<CrossLayerTrack>>,
    bounce_track: Query<'w, 's, (&'static Interaction, &'static ComputedNode, &'static GlobalTransform), With<BouncesTrack>>,
    passes_label: Query<'w, 's, &'static mut Text, (With<BlurPassesLabel>, Without<CrossLayerLabel>, Without<BouncesLabel>)>,
    layer_label: Query<'w, 's, &'static mut Text, (With<CrossLayerLabel>, Without<BlurPassesLabel>, Without<BouncesLabel>)>,
    bounce_label: Query<'w, 's, &'static mut Text, (With<BouncesLabel>, Without<BlurPassesLabel>, Without<CrossLayerLabel>)>,
    passes_fill: Query<'w, 's, &'static mut Node, (With<PassesFill>, Without<CrossLayerFill>, Without<BouncesFill>)>,
    layer_fill: Query<'w, 's, &'static mut Node, (With<CrossLayerFill>, Without<PassesFill>, Without<BouncesFill>)>,
    bounce_fill: Query<'w, 's, &'static mut Node, (With<BouncesFill>, Without<PassesFill>, Without<CrossLayerFill>)>,
}
fn slider_system(
    mut q: SliderQueries,
    windows: Query<&Window>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut blend_toggle: ResMut<BlendToggleState>,
    mut is_dragging_passes: Local<bool>,
    mut is_dragging_layer: Local<bool>,
    mut is_dragging_bounce: Local<bool>,
) {
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let passes_step_size = if shift { 5 } else { 1 };
    let layer_step_size = if shift { 10 } else { 5 }; // 0.5m default, 1.0m with shift

    // -------------------------------------------------------------
    // 1. Diffuse Bounces Slider
    // -------------------------------------------------------------
    let current_bounces = BOUNCE_COUNT.load(Ordering::Relaxed).clamp(1, MAX_BOUNCE_COUNT);
    let mut bounce_delta: i32 = 0;

    for (interaction, step) in &mut q.bounce_buttons {
        if *interaction == Interaction::Pressed {
            bounce_delta += step.0;
        }
    }

    if keys.just_pressed(KeyCode::Digit1) { bounce_delta = 1 - current_bounces as i32; }
    if keys.just_pressed(KeyCode::Digit2) { bounce_delta = 2 - current_bounces as i32; }
    if keys.just_pressed(KeyCode::Digit3) { bounce_delta = 3 - current_bounces as i32; }
    if keys.just_pressed(KeyCode::Digit4) { bounce_delta = 4 - current_bounces as i32; }

    let mut direct_target_bounces: Option<u32> = None;
    if let Ok((interaction, node, transform)) = q.bounce_track.single() {
        if *interaction == Interaction::Pressed {
            *is_dragging_bounce = true;
        }
        if mouse_button.just_released(MouseButton::Left) {
            *is_dragging_bounce = false;
        }

        if *is_dragging_bounce {
            if let Ok(window) = windows.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    let width = node.size().x;
                    if width > 0.0 {
                        let left = transform.translation().x - width * 0.5;
                        let ratio = ((cursor_pos.x - left) / width).clamp(0.0, 1.0);
                        let target = 1 + (ratio * (MAX_BOUNCE_COUNT - 1) as f32).round() as u32;
                        direct_target_bounces = Some(target);
                    }
                }
            }
        }
    }

    let new_bounces = if let Some(target) = direct_target_bounces {
        target.clamp(1, MAX_BOUNCE_COUNT)
    } else if bounce_delta != 0 {
        ((current_bounces as i32) + bounce_delta).clamp(1, MAX_BOUNCE_COUNT as i32) as u32
    } else {
        current_bounces
    };

    if new_bounces != current_bounces || blend_toggle.bounces != new_bounces {
        BOUNCE_COUNT.store(new_bounces, Ordering::Relaxed);
        blend_toggle.bounces = new_bounces;

        for mut text in &mut q.bounce_label {
            text.0 = format!("GI Bounces: {}", new_bounces);
        }

        let pct = (new_bounces.saturating_sub(1)) as f32 / (MAX_BOUNCE_COUNT - 1) as f32 * 100.0;
        for mut fill in &mut q.bounce_fill {
            fill.width = Val::Percent(pct);
        }
    }

    // -------------------------------------------------------------
    // 2. Passes Slider
    // -------------------------------------------------------------
    let current_passes = LOWPASS_BLUR_PASSES
        .load(Ordering::Relaxed)
        .clamp(1, MAX_BLUR_PASSES as u32);
    let mut passes_delta: i32 = 0;

    for (interaction, step) in &mut q.passes_buttons {
        if *interaction == Interaction::Pressed {
            passes_delta += step.0 * passes_step_size;
        }
    }

    if keys.just_pressed(KeyCode::BracketLeft)
        || keys.just_pressed(KeyCode::Minus)
        || keys.just_pressed(KeyCode::NumpadSubtract)
    {
        passes_delta -= passes_step_size;
    }
    if keys.just_pressed(KeyCode::BracketRight)
        || keys.just_pressed(KeyCode::Equal)
        || keys.just_pressed(KeyCode::NumpadAdd)
    {
        passes_delta += passes_step_size;
    }

    let mut direct_target_passes: Option<u32> = None;
    if let Ok((interaction, node, transform)) = q.passes_track.single() {
        if *interaction == Interaction::Pressed {
            *is_dragging_passes = true;
        }
        if mouse_button.just_released(MouseButton::Left) {
            *is_dragging_passes = false;
        }

        if *is_dragging_passes {
            if let Ok(window) = windows.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    let width = node.size().x;
                    if width > 0.0 {
                        let left = transform.translation().x - width * 0.5;
                        let ratio = ((cursor_pos.x - left) / width).clamp(0.0, 1.0);
                        let target = 1 + (ratio * (MAX_BLUR_PASSES - 1) as f32).round() as u32;
                        direct_target_passes = Some(target);
                    }
                }
            }
        }
    }

    let new_passes = if let Some(target) = direct_target_passes {
        target.clamp(1, MAX_BLUR_PASSES as u32)
    } else if passes_delta != 0 {
        ((current_passes as i32) + passes_delta).clamp(1, MAX_BLUR_PASSES as i32) as u32
    } else {
        current_passes
    };

    if new_passes != current_passes || blend_toggle.blur_passes != new_passes {
        LOWPASS_BLUR_PASSES.store(new_passes, Ordering::Relaxed);
        blend_toggle.blur_passes = new_passes;

        for mut text in &mut q.passes_label {
            text.0 = format!("Blur Passes: {}", new_passes);
        }

        let pct = (new_passes.saturating_sub(1)) as f32 / (MAX_BLUR_PASSES - 1) as f32 * 100.0;
        for mut fill in &mut q.passes_fill {
            fill.width = Val::Percent(pct);
        }
    }

    // -------------------------------------------------------------
    // 3. Cross-Layer Radius Slider
    // -------------------------------------------------------------
    let current_radius_tenths = CROSS_LAYER_RADIUS_TENTHS
        .load(Ordering::Relaxed)
        .clamp(0, MAX_CROSS_LAYER_TENTHS);
    let mut radius_delta: i32 = 0;

    for (interaction, step) in &mut q.layer_buttons {
        if *interaction == Interaction::Pressed {
            radius_delta += step.0 * layer_step_size;
        }
    }

    if keys.just_pressed(KeyCode::Comma) {
        radius_delta -= layer_step_size;
    }
    if keys.just_pressed(KeyCode::Period) {
        radius_delta += layer_step_size;
    }

    let mut direct_target_radius: Option<u32> = None;
    if let Ok((interaction, node, transform)) = q.layer_track.single() {
        if *interaction == Interaction::Pressed {
            *is_dragging_layer = true;
        }
        if mouse_button.just_released(MouseButton::Left) {
            *is_dragging_layer = false;
        }

        if *is_dragging_layer {
            if let Ok(window) = windows.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    let width = node.size().x;
                    if width > 0.0 {
                        let left = transform.translation().x - width * 0.5;
                        let ratio = ((cursor_pos.x - left) / width).clamp(0.0, 1.0);
                        let target = (ratio * MAX_CROSS_LAYER_TENTHS as f32).round() as u32;
                        direct_target_radius = Some(target);
                    }
                }
            }
        }
    }

    let new_radius_tenths = if let Some(target) = direct_target_radius {
        target.clamp(0, MAX_CROSS_LAYER_TENTHS)
    } else if radius_delta != 0 {
        ((current_radius_tenths as i32) + radius_delta).clamp(0, MAX_CROSS_LAYER_TENTHS as i32) as u32
    } else {
        current_radius_tenths
    };

    let new_radius_m = new_radius_tenths as f32 / 10.0;
    if new_radius_tenths != current_radius_tenths || (blend_toggle.cross_layer_radius - new_radius_m).abs() > 0.01 {
        CROSS_LAYER_RADIUS_TENTHS.store(new_radius_tenths, Ordering::Relaxed);
        blend_toggle.cross_layer_radius = new_radius_m;

        for mut text in &mut q.layer_label {
            text.0 = if new_radius_m <= 0.001 {
                "Cross-Layer: 0.0 m (isolated)".to_string()
            } else {
                format!("Cross-Layer Radius: {:.1} m", new_radius_m)
            };
        }

        let pct = (new_radius_tenths as f32 / MAX_CROSS_LAYER_TENTHS as f32) * 100.0;
        for mut fill in &mut q.layer_fill {
            fill.width = Val::Percent(pct);
        }
    }
}
