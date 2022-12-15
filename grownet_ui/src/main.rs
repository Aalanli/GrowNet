#![allow(dead_code)]

use bevy::{prelude::*, asset::AssetServerSettings};
//use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_stl;
use bevy_egui::EguiPlugin;

use grownet_lib as lib;
use lib::ui;

use lib::data_configs;

/// the path at which the user config files are stored
const ROOT_PATH: &str = "assets/config";


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
        //.add_plugin(WorldInspectorPlugin::new())
        .add_plugin(bevy_stl::StlPlugin)
        .add_system(bevy::window::close_when_requested)
        .add_startup_system_to_stage(StartupStage::PreStartup, setup_ui)
        .add_startup_system_to_stage(StartupStage::PreStartup, setup_dataset_ui)
        .add_plugin(ui::UIPlugin)
        .run();
}

fn setup_ui(mut commands: Commands) {
    let mut params = ui::UIParams::default();
    params.root_path = ROOT_PATH.to_string();
    commands.insert_resource(params);
}

fn setup_dataset_ui(mut commands: Commands) {
    use lib::datasets as data;
    let mut dataset_ui = ui::DatasetUI::default();

    dataset_ui.push_viewer(data_configs::build_viewer(), "cifar10Torchimpl");
    let cifar10 = data::cifar::Cifar10Params::default();
    let cifar_viewer = ui::data_ui::ClassificationViewerOwned::new(cifar10);
    dataset_ui.push_viewer(cifar_viewer, "cifar10");

    let mnist = data::mnist::MnistParams::default();
    let mnist_viewer = ui::data_ui::ClassificationViewerOwned::new(mnist);
    dataset_ui.push_viewer(mnist_viewer, "mnist");

    commands.insert_resource(dataset_ui);
}