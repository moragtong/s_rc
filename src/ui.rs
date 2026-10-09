use bevy::prelude::*;

pub struct UiPlugin;

#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]
pub struct LightingState {
    pub indirect_enabled: bool,
    pub probes_enabled: bool,
    pub trilinear_enabled: bool,
    pub bilateral_enabled: bool,
    pub normal_bias_enabled: bool,
}

impl Default for LightingState {
    fn default() -> Self {
        Self {
            indirect_enabled: true,
            probes_enabled: false,
            trilinear_enabled: true,
            bilateral_enabled: false,
            // On: the fetch is pushed along the normal into the cell in front
            // of the surface and `probe_visible` is skipped (the normal bias is
            // meant to replace the trace, see `mytodo.md` section 4.3). Off:
            // the probe grid is read at the shading point and `probe_visible`
            // does the occlusion rejection. The two are alternatives, hence the
            // toggle.
            normal_bias_enabled: true,
        }
    }
}

#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]
pub struct ParallaxState {
    pub enabled: bool,
}

impl Default for ParallaxState {
    // Off since the two-layer probe model: the endpoint parallax the toggle
    // switches was a correction for single-layer mid-room probes, and the
    // layers removed the need for it. Kept so the old behaviour can be A/B'd.
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Clone, Copy, bevy::render::extract_resource::ExtractResource)]
pub struct CrossLayerState {
    pub enabled: bool,
}

impl Default for CrossLayerState {
    // Off: merges are per-layer. When on, each cascade child merges into the
    // other layer's parents (floor child -> ceiling parents and vice versa).
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Component)]
pub struct ActionButton(pub Action);

#[derive(Clone, Copy)]
pub enum Action {
    ToggleHeightmap,
    ToggleIndirect,
    ToggleProbes,
    ToggleParallax,
    ToggleTrilinear,
    ToggleCrossLayer,
    ToggleBilateral,
    ToggleNormalBias,
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightingState>()
            .init_resource::<ParallaxState>()
            .init_resource::<CrossLayerState>()
            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                LightingState,
            >::default())
            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                ParallaxState,
            >::default())
            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                CrossLayerState,
            >::default())
            .add_systems(Startup, setup_ui)
            .add_systems(Update, button_system);
    }
}

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(10.0),
            ..default()
        },))
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
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
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
                    ActionButton(Action::ToggleParallax),
                ))
                .with_child((
                    Text::new("Parallax Endpoint (O)"),
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
                    ActionButton(Action::ToggleTrilinear),
                ))
                .with_child((
                    Text::new("Trilinear Shade (T)"),
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
                    ActionButton(Action::ToggleCrossLayer),
                ))
                .with_child((
                    Text::new("Cross-Layer Merge (L)"),
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
                    ActionButton(Action::ToggleBilateral),
                ))
                .with_child((
                    Text::new("Bilateral Filter (B)"),
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
                    ActionButton(Action::ToggleNormalBias),
                ))
                .with_child((
                    Text::new("Normal-Biased Fetch (N)"),
                    TextColor(Color::WHITE),
                ));
        });
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &ActionButton),
        With<Button>,
    >,
    keys: Res<ButtonInput<KeyCode>>,
    mut heightmap_query: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut lighting_state: ResMut<LightingState>,
    mut parallax_state: ResMut<ParallaxState>,
    mut cross_layer_state: ResMut<CrossLayerState>,
) {
    let mut toggle_h = keys.just_pressed(KeyCode::KeyM);
    let mut toggle_i = keys.just_pressed(KeyCode::KeyI);
    let mut toggle_p = keys.just_pressed(KeyCode::KeyP);
    let mut toggle_par = keys.just_pressed(KeyCode::KeyO);
    let mut toggle_tri = keys.just_pressed(KeyCode::KeyT);
    let mut toggle_cross = keys.just_pressed(KeyCode::KeyL);
    let mut toggle_bi = keys.just_pressed(KeyCode::KeyB);
    let mut toggle_bias = keys.just_pressed(KeyCode::KeyN);

    for (interaction, mut color, action) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = BackgroundColor(Color::srgb(0.35, 0.75, 0.35));
                match action.0 {
                    Action::ToggleHeightmap => toggle_h = true,
                    Action::ToggleIndirect => toggle_i = true,
                    Action::ToggleProbes => toggle_p = true,
                    Action::ToggleParallax => toggle_par = true,
                    Action::ToggleTrilinear => toggle_tri = true,
                    Action::ToggleCrossLayer => toggle_cross = true,
                    Action::ToggleBilateral => toggle_bi = true,
                    Action::ToggleNormalBias => toggle_bias = true,
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

    if toggle_par {
        parallax_state.enabled = !parallax_state.enabled;
        bevy::log::info!("Parallax endpoint: {}", parallax_state.enabled);
    }

    if toggle_tri {
        lighting_state.trilinear_enabled = !lighting_state.trilinear_enabled;
        bevy::log::info!("Trilinear shade: {}", lighting_state.trilinear_enabled);
    }

    if toggle_cross {
        cross_layer_state.enabled = !cross_layer_state.enabled;
        bevy::log::info!(
            "Cross-layer merge: {}",
            if cross_layer_state.enabled {
                "on (the other layer's parents)"
            } else {
                "off (per-layer)"
            }
        );
    }

    if toggle_bi {
        lighting_state.bilateral_enabled = !lighting_state.bilateral_enabled;
        bevy::log::info!("Bilateral filter: {}", lighting_state.bilateral_enabled);
    }

    if toggle_bias {
        lighting_state.normal_bias_enabled = !lighting_state.normal_bias_enabled;
        bevy::log::info!("Normal-biased fetch: {}", lighting_state.normal_bias_enabled);
    }

    // The button colour is the on-screen indicator: green while the toggle is
    // enabled, dark grey while it is disabled.
    for (_interaction, mut color, action) in &mut interaction_query {
        match action.0 {
            Action::ToggleParallax => {
                *color = BackgroundColor(if parallax_state.enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            Action::ToggleTrilinear => {
                *color = BackgroundColor(if lighting_state.trilinear_enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            Action::ToggleBilateral => {
                *color = BackgroundColor(if lighting_state.bilateral_enabled {
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
            Action::ToggleCrossLayer => {
                *color = BackgroundColor(if cross_layer_state.enabled {
                    Color::srgb(0.35, 0.75, 0.35)
                } else {
                    Color::srgb(0.15, 0.15, 0.15)
                });
            }
            _ => {}
        }
    }
}
