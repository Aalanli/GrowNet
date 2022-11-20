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
use anyhow::{Result, Error, Context};
use serde::{Serialize, Deserialize};

use core::hash::Hash;

mod dataset_ui;
use dataset_ui::DatasetState;
use crate::datasets::{DatasetEnum, DatasetTypes, DatasetBuilder, mnist};

/// the path at which the user config files are stored
const ROOT_PATH: &str = "assets/config";

pub trait Param {
    fn ui(&mut self, ui: &mut egui::Ui);
    fn config(&self) -> String;
    fn load_config(&mut self, config: &str);
}

fn deserialize_hashmap<'de, T, K: Param + Default>(a: &'de str) -> Result<HashMap<T, K>>
where T: Deserialize<'de> + std::cmp::Eq + std::hash::Hash + Clone 
{
    let temp: HashMap<T, String> = ron::from_str(a).with_context(|| {"failed to deserialize hashmap"})?;
    let t = temp.iter().map(|(x, y)| {
        let mut h = K::default();
        h.load_config(&y);
        (x.clone(), h)
    }).collect();
    Ok(t)
}

fn serialize_hashmap<T, K: Param>(map: &HashMap<T, K>) -> Result<String> 
where T: Serialize + std::cmp::Eq + std::hash::Hash + Clone 
{
    let temp: HashMap<T, String> = map.iter().map(|(x, y)| {(x.clone(), y.config())}).collect();
    ron::to_string(&temp).with_context(|| "failed to serialize hashmap")
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
    font_delta: f32,
}

/// State for panel opened in the ui
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
enum OpenPanel {
    Models,
    Datasets,
    Misc,
    Train
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
            OpenPanel::Datasets => {dataset_state.ui(ui)},
            OpenPanel::Misc     => {params.update_misc(ui)},
            OpenPanel::Train => {},
        }
    });
}

fn setup_ui(mut commands: Commands, mut egui_context: ResMut<EguiContext>) {
    let mut params = UIParams::default();
    let mut dataset_state = DatasetState::default();

    let root_path: path::PathBuf = ROOT_PATH.into();
    let config_file = root_path.join("ui_config").with_extension("ron");
    if config_file.exists() {
        let config: (String, String) = ron::from_str(&fs::read_to_string(&config_file).unwrap()).unwrap();
        params.load_config(&config.0);
        dataset_state.load_config(&config.1);
    }

    // startup tasks that one must do to update the ui
    change_font_size(params.font_delta, egui_context.ctx_mut());

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
        let root_path: path::PathBuf = ROOT_PATH.into();
        if !root_path.exists() {
            fs::create_dir_all(&root_path).unwrap();
        }

        eprintln!("saving config");
        let config_file = root_path.join("ui_config").with_extension("ron");
        let main_ui_config = params.config();
        let data_ui_config = dataset_state.config();
        let serialized = ron::to_string(&(main_ui_config, data_ui_config)).unwrap();
        fs::write(&config_file, serialized).unwrap();
    }
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
        UIParams { open_panel: OpenPanel::Models, font_delta: 4.0 }
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

