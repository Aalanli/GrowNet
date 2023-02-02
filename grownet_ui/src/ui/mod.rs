use core::hash::Hash;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs;
use std::path;

use anyhow::{Context, Result};
use bincode;
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::window::{WindowCloseRequested, WindowClosed};
use bevy_egui::{egui, EguiContext};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{Configure, UI};

pub mod data_ui;
pub mod train_ui;

/// the path at which the user config files are stored
const ROOT_PATH: &str = "assets/config";

/// The ui plugin, the entry point for the ui
pub struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(UIParams::default())
            .add_startup_system_to_stage(StartupStage::Startup,setup_ui)
            .add_system(save_ui)
            .add_state(AppState::Menu)
            .add_plugin(data_ui::DatasetUIPlugin)
            .add_plugin(train_ui::TrainUIPlugin)
            .add_system_set(SystemSet::on_update(AppState::Menu).with_system(menu_ui))
            ;
    }
}

fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
    mut dataset_state: ResMut<data_ui::DatasetUI>,
    mut train_state: ResMut<train_ui::TrainingUI>,
    mut train_env: ResMut<train_ui::TrainEnviron>,
    mut app_state: ResMut<State<AppState>>,
) {
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));

        ui.horizontal(|ui| {
            // The four possible states for the ui to be in,
            // selecting "Train" switches to the Trainer app state
            ui.selectable_value(&mut params.open_panel, OpenPanel::Models, "Models");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Datasets, "Datasets");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Misc, "Misc");
        });
        ui.separator();

        match params.open_panel {
            OpenPanel::Models => { train_state.ui(ui, &mut train_env, &mut *app_state); }
            OpenPanel::Datasets => dataset_state.ui(ui),
            OpenPanel::Misc => params.update_misc(ui),
        }
    });
}

fn setup_ui(
    mut params: ResMut<UIParams>,
    mut egui_context: ResMut<EguiContext>,
) {
    params.root_path = ROOT_PATH.to_string();
    
    let root_path: path::PathBuf = params.root_path.clone().into();
    let config_file = root_path.join("ui_config").with_extension("ron");
    // loading configurations of main ui components
    if config_file.exists() {
        eprintln!("loading from config file {}", config_file.to_str().unwrap());
        let result: Result<String, _> = ron::from_str(&fs::read_to_string(&config_file).unwrap());
        match result {
            Ok(config) => { params.load_config(&config); },
            Err(e) => { eprintln!("unable to deserialize ui_params {}", e); }
        }
    }

    // startup tasks that one must do to update the ui
    change_font_size(params.font_delta, egui_context.ctx_mut());

}

fn save_ui(
    mut exit: EventReader<AppExit>,
    mut closed: EventReader<WindowClosed>,
    mut closed2: EventReader<WindowCloseRequested>,
    mut app_state: ResMut<State<AppState>>,
    params: Res<UIParams>,
) {
    let mut exited = false;
    for _ in exit.iter() {
        exited = true;
    }
    for _ in closed.iter() {
        exited = true;
    }
    for _ in closed2.iter() {
        exited = true;
    }

    if exited {
        let root_path: path::PathBuf = params.root_path.clone().into();
        if !root_path.exists() {
            fs::create_dir_all(&root_path).unwrap();
        }

        eprintln!("saving ui config");
        let config_file = root_path.join("ui_config").with_extension("ron");

        let main_ui_config = params.config();

        let serialized = ron::to_string(&main_ui_config).unwrap();
        fs::write(&config_file, serialized).unwrap();

        app_state.set(AppState::Close).expect("failed to send app close msg");
    }
}

/// Main configuration state for the entire ui
/// as in, the ui can be constructed solely from these parameters
#[derive(Debug, Resource, Serialize, Deserialize)]
pub struct UIParams {
    pub root_path: String,
    pub font_delta: f32,
    open_panel: OpenPanel,
}

/// The state for the entire app, which characterizes the two main modes of operation
/// Menu involves only light ui tasks, while Trainer may involve some heavy compute,
/// (which may run on another thread), and rendering.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum AppState {
    Menu,
    Trainer,
    Close,
}

/// State for panel opened in the ui
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
enum OpenPanel {
    Models,
    Datasets,
    Misc,
}

impl UIParams {
    fn load_config(&mut self, config: &str) {
        *self = ron::from_str(config).unwrap()
    }

    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn update_misc(&mut self, ui: &mut egui::Ui) {
        let mut local_font_delta = self.font_delta;
        // stylistic changes

        ui.label("font size delta");
        ui.add(egui::Slider::new(&mut local_font_delta, -9.0..=12.0));
        ui.end_row();

        if local_font_delta != self.font_delta {
            change_font_size(local_font_delta, ui.ctx());
            self.font_delta = local_font_delta;
        }
    }
}

impl Default for UIParams {
    fn default() -> Self {
        UIParams {
            open_panel: OpenPanel::Models,
            font_delta: 4.0,
            root_path: "".to_string(),
        }
    }
}


fn change_font_size(font_delta: f32, ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(18.0 + font_delta, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(10.0 + font_delta, egui::FontFamily::Proportional),
    );
    ctx.set_style(style);
}
