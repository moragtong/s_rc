use bevy::prelude::*;

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
}

impl Default for BlendToggleState {
    fn default() -> Self {
        Self {
            lowpass: true,
            blur_passes: 1,
        }
    }
}

#[derive(Component)]
pub struct ActionButton(pub Action);

#[derive(Clone, Copy)]
pub enum Action {
    ToggleHeightmap,
    ToggleIndirect,
    ToggleProbes,
    ToggleVisMask,
    ToggleNormalBias,
    ToggleLowPass,
    ToggleSsoA,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightingState>()
            .init_resource::<BlendToggleState>()
            .add_plugins((
                bevy::render::extract_resource::ExtractResourcePlugin::<LightingState>::default(),
                bevy::render::extract_resource::ExtractResourcePlugin::<BlendToggleState>::default(
                ),
            ))
            .add_systems(Startup, setup_ui)
            .add_systems(Update, button_system);
    }
}

/// Draws an on-screen button bar along the top edge of the window.
///
/// Designed so screenshots can crop or ignore the top 48 pixels and get an
/// A/B diff (`S_RC_INDIRECT`-style comparisons are on the scene, not the UI);
/// the hotkeys (M, I, P...) do the exact same work as clicking the buttons.
fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                left: Val::Px(8.0),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(6.0),
                ..default()
            },
            // Pick up clicks through the rest of the UI root so the scene
            // camera doesn't steal them.
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
                .with_child((Text::new("Heightmap (M)"), TextColor(Color::WHITE)));
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
                .with_child((Text::new("Indirect (I)"), TextColor(Color::WHITE)));
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
                .with_child((Text::new("Probes (P)"), TextColor(Color::WHITE)));
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

    for (interaction, mut color, action) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = BackgroundColor(Color::srgb(0.35, 0.75, 0.35));
                match action.0 {
                    Action::ToggleHeightmap => toggle_h = true,
                    Action::ToggleIndirect => toggle_i = true,
                    Action::ToggleProbes => toggle_p = true,
                    Action::ToggleVisMask => toggle_vis = true,
                    Action::ToggleNormalBias => toggle_bias = true,
                    Action::ToggleLowPass => toggle_lp = true,
                    Action::ToggleSsoA => toggle_ssao = true,
                }
            }
            Interaction::Hovered => {
                *color = BackgroundColor(Color::srgb(0.25, 0.25, 0.25));
            }
            Interaction::None => {
                *color = BackgroundColor(Color::srgb(0.15, 0.15, 0.15));
            }
        }
    }

    if toggle_h {
        for mut vis in &mut heightmap_query {
            *vis = match *vis {
                Visibility::Hidden => Visibility::Inherited,
                _ => Visibility::Hidden,
            };
        }
    }

    if toggle_i {
        lighting_state.indirect_enabled = !lighting_state.indirect_enabled;
        bevy::log::info!("Indirect Lighting: {}", lighting_state.indirect_enabled);
    }

    if toggle_p {
        lighting_state.probes_enabled = !lighting_state.probes_enabled;
    }

    if toggle_vis {
        lighting_state.use_vis_mask = !lighting_state.use_vis_mask;
        bevy::log::info!(
            "Visibility test mode: {}",
            if lighting_state.use_vis_mask {
                "probe_is_visible (precalculated texture mask)"
            } else {
                "probe_visible (per-tap ray trace)"
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
        bevy::log::info!("Low-pass filter: {}", blend_toggle.lowpass);
    }

    if toggle_ssao {
        lighting_state.bevy_ssao_enabled = !lighting_state.bevy_ssao_enabled;
        bevy::log::info!("Bevy SSAO: {}", lighting_state.bevy_ssao_enabled);
    }

    // The button colour is the on-screen indicator: green while the toggle is
    // enabled, dark grey while it is disabled.
    for (_interaction, mut color, action) in &mut interaction_query {
        match action.0 {
            Action::ToggleVisMask => {
                *color = BackgroundColor(if lighting_state.use_vis_mask {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            Action::ToggleNormalBias => {
                *color = BackgroundColor(if lighting_state.normal_bias_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            Action::ToggleLowPass => {
                *color = BackgroundColor(if blend_toggle.lowpass {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            Action::ToggleSsoA => {
                *color = BackgroundColor(if lighting_state.bevy_ssao_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            _ => {}
        }
    }
}
