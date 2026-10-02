use bevy::prelude::*;

pub struct UiPlugin;

#[derive(Resource)]
pub struct LightingState {
    pub indirect_enabled: bool,
}

impl Default for LightingState {
    fn default() -> Self {
        Self { indirect_enabled: true }
    }
}

#[derive(Component)]
pub struct ActionButton(pub Action);

#[derive(Clone, Copy)]
pub enum Action {
    ToggleHeightmap,
    ToggleIndirect,
}

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightingState>()
            .add_systems(Startup, setup_ui)
            .add_systems(Update, button_system);
    }
}

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(10.0),
                ..default()
            },
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
                .with_child((
                    Text::new("Toggle Heightmap (M)"),
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
                    ActionButton(Action::ToggleIndirect),
                ))
                .with_child((
                    Text::new("Toggle Indirect Lighting (I)"),
                    TextColor(Color::WHITE),
                ));
        });
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &ActionButton),
        (Changed<Interaction>, With<Button>),
    >,
    keys: Res<ButtonInput<KeyCode>>,
    mut heightmap_query: Query<&mut Visibility, With<crate::heightmap_view::HeightmapViewMesh>>,
    mut lighting_state: ResMut<LightingState>,
) {
    let mut toggle_h = keys.just_pressed(KeyCode::KeyM);
    let mut toggle_i = keys.just_pressed(KeyCode::KeyI);

    for (interaction, mut color, action) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = BackgroundColor(Color::srgb(0.35, 0.75, 0.35));
                match action.0 {
                    Action::ToggleHeightmap => toggle_h = true,
                    Action::ToggleIndirect => toggle_i = true,
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
}
