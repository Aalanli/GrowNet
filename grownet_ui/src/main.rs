#![allow(dead_code)]

use bevy::prelude::*;
//use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_egui::EguiPlugin;
use bevy_stl;

use grownet_lib as lib;
use lib::ui;

fn main() {
    App::new()
        .insert_resource(TaskPoolOptions {
            max_total_threads: 1,
            ..default()
        })
        .add_plugins(
            DefaultPlugins.set(AssetPlugin {
                watch_for_changes: true,
                ..default()
            }).set(WindowPlugin {
                exit_on_all_closed: false, // we want to clean up any processes, so don't exit immediately
                close_when_requested: false, // only close window when all processes are killed
                ..default()
            })
        )
        .add_plugin(EguiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
        .add_plugin(bevy_stl::StlPlugin)
        .add_plugin(ui::UIPlugin)
        .run();
}
