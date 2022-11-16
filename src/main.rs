#![allow(dead_code)]

use bevy::{prelude::*, asset::AssetServerSettings};
use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_stl;
use bevy_egui::EguiPlugin;
use grownet_lib::ui;
use grownet_lib::visualizations::renderer::render3d::SimpleViewer;
fn main() {
    App::new()
        .insert_resource(DefaultTaskPoolOptions {
                max_total_threads: 1,
                ..default()
        })
        .insert_resource(AssetServerSettings {
            watch_for_changes: true,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_plugin(bevy_stl::StlPlugin)
        .add_plugin(ui::UI)
        .run();
}
