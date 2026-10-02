sed -i 's/use bevy::prelude::\*/use bevy::prelude::\*;\nuse bevy::hierarchy::ChildBuilder;/g' src/ui.rs
cargo check
