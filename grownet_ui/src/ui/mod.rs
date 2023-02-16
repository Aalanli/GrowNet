use core::hash::Hash;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs;
use std::path;

use anyhow::{Context, Result};
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::window::{WindowCloseRequested, WindowClosed};
use bevy_egui::{egui, EguiContext};
use bincode;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{Configure, UI};

pub mod data_ui;
pub mod train_ui;

/// The ui plugin, the entry point for the ui
pub struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(UIParams::default())
            .add_startup_system_to_stage(StartupStage::Startup, setup_ui)
            .add_state(AppState::Models)
            .add_state(OperatingState::Active)
            .add_plugin(data_ui::DatasetUIPlugin)
            .add_plugin(train_ui::TrainUIPlugin)
            .add_system_set(SystemSet::on_update(AppState::Menu).with_system(menu_ui))
            .add_system_set(SystemSet::on_update(OperatingState::Active).with_system(should_cleanup))
            .add_system_set(SystemSet::on_update(OperatingState::Close)
                .with_system(save_ui)
                .with_system(close_ui)); // final bevy cleanup
    }
}

fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
    mut dataset_state: ResMut<data_ui::DatasetUI>,
    mut app_state: ResMut<State<AppState>>,
    op_state: ResMut<State<OperatingState>>,
) {
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));

        let prev_panel = params.open_panel;
        handle_pane_options(ui, &mut params.open_panel);

        match params.open_panel {
            OpenPanel::Models => {
                app_state.set(AppState::Models).unwrap();
            }
            OpenPanel::Datasets => dataset_state.ui(ui),
            OpenPanel::Misc => params.update_misc(ui, op_state), // force kill option
            OpenPanel::Trainer => {
                // stupid hack, as if open_panel is ever Trainer, then the training menu system will get stuck trying to go back
                params.open_panel = prev_panel;
                app_state.set(AppState::Trainer).unwrap()
            }
        }
    });
}

/// The heading pane in the ui
fn handle_pane_options(ui: &mut egui::Ui, panel: &mut OpenPanel) {
    ui.horizontal(|ui| {
        // The three possible states for the ui to be in,
        // selecting "Train" switches to the Trainer app state
        ui.selectable_value(panel, OpenPanel::Models, "Models");
        ui.selectable_value(panel, OpenPanel::Datasets, "Datasets");
        ui.selectable_value(panel, OpenPanel::Misc, "Misc");
        ui.selectable_value(panel, OpenPanel::Trainer, "Train Environment");
    });
    ui.separator();
}

fn setup_ui(mut params: ResMut<UIParams>, mut egui_context: ResMut<EguiContext>) {
    let root_path: path::PathBuf = crate::CONFIG_PATH.into();

    let config_file = root_path.join("ui_config").with_extension("ron");
    // loading configurations of main ui components
    if config_file.exists() {
        eprintln!("loading from config file {}", config_file.to_str().unwrap());
        let result: Result<String, _> = ron::from_str(&fs::read_to_string(&config_file).unwrap());
        match result {
            Ok(config) => {
                params.load_config(&config);
            }
            Err(e) => {
                eprintln!("unable to deserialize ui_params {}", e);
            }
        }
    }

    // startup tasks that one must do to update the ui
    change_font_size(params.font_delta, egui_context.ctx_mut());
}

fn save_ui(
    params: Res<UIParams>,
) {
    let root_path: path::PathBuf = crate::CONFIG_PATH.into();
    if !root_path.exists() {
        fs::create_dir_all(&root_path).unwrap();
    }

    eprintln!("saving ui config");
    let config_file = root_path.join("ui_config").with_extension("ron");

    let main_ui_config = params.config();

    let serialized = ron::to_string(&main_ui_config).unwrap();
    fs::write(&config_file, serialized).unwrap();
    
}

/// cleanup when user tries to close the window
fn should_cleanup(
    mut close: EventReader<WindowCloseRequested>,
    mut app_state: ResMut<State<OperatingState>>
) {
    for _ in close.iter() {
        app_state.set(OperatingState::Cleanup).expect("unable to set cleanup state");
        break;
    }
}

/// close when cleanup is finished
fn close_ui(
    mut exit: EventWriter<AppExit>,
    windows: ResMut<Windows>,
    closed: EventReader<WindowCloseRequested>,
) {
    bevy::window::close_when_requested(windows, closed);
    exit.send(AppExit);
}

/// Main configuration state for the entire ui
/// as in, the ui can be constructed solely from these parameters
#[derive(Debug, Resource, Serialize, Deserialize)]
pub struct UIParams {
    pub font_delta: f32,
    open_panel: OpenPanel,
    pub run_queue_max_active: usize,
    pub run_queue_num_errs: usize,
}

/// The state for the entire app, which characterizes the two main modes of operation
/// Menu involves only light ui tasks, while Trainer may involve some heavy compute,
/// (which may run on another thread), and rendering.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum AppState {
    Menu,
    Models,
    Trainer
}


#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum OperatingState {
    Active,  // Normal app 
    Cleanup, // Used in the task spawner to kill any processes
    Close,   // final app close, when the task spawner has killed all processes
}


/// State for panel opened in the ui
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize, Copy, Clone)]
enum OpenPanel {
    Trainer,
    Models,
    Datasets,
    Misc,
}

impl UIParams {
    fn load_config(&mut self, config: &str) {
        ron::from_str(config).map_or_else(|err| {
            eprintln!("unable to load UIParams: {}", err.to_string());
        },|c| {
            *self = c;
        });
    }

    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }

    pub fn update_misc(&mut self, ui: &mut egui::Ui, mut state: ResMut<State<OperatingState>>) {
        let mut local_font_delta = self.font_delta;
        // stylistic changes

        ui.label("font size delta");
        ui.add(egui::Slider::new(&mut local_font_delta, -9.0..=12.0));
        ui.end_row();

        if local_font_delta != self.font_delta {
            change_font_size(local_font_delta, ui.ctx());
            self.font_delta = local_font_delta;
        }
        
        ui.label("run queue maximum active runs");
        ui.add(egui::Slider::new(&mut self.run_queue_max_active, 1..=64));

        ui.label("run queue maximum number of error messages");
        ui.add(egui::Slider::new(&mut self.run_queue_num_errs, 1..=100));

        // emergency kill switch, in case some processes are unable to be killed
        if ui.button("force kill").clicked() {
            state.set(OperatingState::Close).unwrap();
        }
    }
}

impl Default for UIParams {
    fn default() -> Self {
        UIParams {
            open_panel: OpenPanel::Models,
            font_delta: 4.0,
            run_queue_max_active: 1,
            run_queue_num_errs: 5,
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
