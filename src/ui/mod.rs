use std::{borrow::Cow, mem::MaybeUninit};
use std::collections::HashMap;
use std::path;
use std::fs;

use bevy::prelude::*;
use bevy::app::AppExit;
use bevy::window::{WindowClosed, WindowCloseRequested};
use bevy_egui::{egui, EguiContext};

use ndarray::{s, Axis};
use strum::IntoEnumIterator;
use anyhow::{Result, Error};
use serde::{Serialize, Deserialize};


mod dataset_ui;
use dataset_ui::DatasetState;
use crate::datasets::{DatasetEnum, DatasetTypes, DatasetUI, mnist};

/// the path at which the user config files are stored
const ROOT_PATH: &str = "assets/config";


pub trait ViewerUI: Send + Sync {
    fn ui(&mut self, ui: &mut egui::Ui);
}

/// The ui plugin, the entry point for the entire ui
pub struct UI;

/// The state for the entire app, which characterizes the two main modes of operation
/// Menu involves only light ui tasks, while Trainer may involve some heavy compute, 
/// (which may run on another thread), and rendering.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
enum State {
    Menu,
    Trainer
}

/// Main configuration state for the entire ui
/// as in, the ui can be constructed solely from these parameters
#[derive(Debug, Serialize, Deserialize)]
struct UIParams {
    open_panel: OpenPanel,
    misc: Misc
}

/// State for panel opened in the ui
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
enum OpenPanel {
    Models,
    Datasets,
    Misc,
    Train
}

/// state for the various miscellaneous configuration settings for the ui
#[derive(Debug, Serialize, Deserialize)]
struct Misc {
    font_delta: f32, // the amount to increase or decrease font size 
}


impl Plugin for UI {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system(setup_ui)
            .add_system(save_ui)
            .add_system(menu_ui);
    }
}

fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
    mut dataset_state: ResMut<DatasetState>,
) {
    
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        ui.add(egui::Label::new("Data Explorer"));

        ui.horizontal(|ui| {
            // The four possible states for the ui to be in,
            // selecting "Train" switches to the Trainer app state
            ui.selectable_value(&mut params.open_panel, OpenPanel::Models, "Models");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Datasets, "Datasets");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Misc, "Misc");
            ui.selectable_value(&mut params.open_panel, OpenPanel::Train, "Train");
        });
        ui.separator();

        match params.open_panel {
            OpenPanel::Models => {},
            OpenPanel::Datasets => {dataset_state.update(ui)},
            OpenPanel::Misc     => {params.misc.update_misc(ui)},
            OpenPanel::Train => {},
        }
    });
}

fn setup_ui(mut commands: Commands,) {
    let mut params = UIParams::default();
    let mut dataset_state = DatasetState::default();

    let root_path: path::PathBuf = ROOT_PATH.into();
    params.setup(&root_path);
    dataset_state.setup(&root_path);

    commands.insert_resource(params);
    commands.insert_resource(dataset_state);
}

fn save_ui(
    mut exit: EventReader<AppExit>, 
    mut closed: EventReader<WindowClosed>, 
    mut closed2: EventReader<WindowCloseRequested>,
    params: Res<UIParams>,
    dataset_state: Res<DatasetState>
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
        eprintln!("saving config");
        let root_path: path::PathBuf = ROOT_PATH.into();
        params.save_params(&root_path);
        dataset_state.save_params(&root_path);
    }
}

impl UIParams {
    fn setup(&mut self, root_path: &path::Path) {
        let ui_path = root_path.join("ui_config").with_extension("ron");
        if ui_path.exists() {
            let config = fs::read_to_string(&ui_path).unwrap();
            *self = ron::from_str(&config).unwrap()
        }
    }

    fn save_params(&self, root_path: &path::Path) {
        let ui_path = root_path.join("ui_config").with_extension("ron");
        let config_str = ron::to_string(self).unwrap();
        if !ui_path.parent().unwrap().exists() {
            fs::create_dir_all(&ui_path.parent().unwrap()).unwrap();
        }
        fs::write(&ui_path, config_str).unwrap();
        
    }
}

impl Default for UIParams {
    fn default() -> Self {
        UIParams { misc: Misc::default(), open_panel: OpenPanel::Models }
    }
}

impl Misc {
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

impl Default for Misc {
    fn default() -> Self {
        Misc { 
            font_delta: 4.0 
        }
    }
}

fn change_font_size(font_delta: f32, ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::new(18.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Monospace, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::new(10.0 + font_delta, egui::FontFamily::Proportional));
    ctx.set_style(style);
}

