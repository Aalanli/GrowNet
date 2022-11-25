use core::hash::Hash;
use std::fmt::Display;
use std::collections::HashMap;
use std::path;
use std::fs;

use anyhow::Context;
use bevy::prelude::*;
use bevy::app::AppExit;
use bevy::window::{WindowClosed, WindowCloseRequested};
use bevy_egui::{egui, EguiContext};

use anyhow::Result;
use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::{Param, Config, UI};

/// Defines everything that gets the dataset viewer to work
pub mod data_ui;
pub use data_ui::DatasetUI;

/// The ui plugin, the entry point for the ui
pub struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_startup_system_to_stage(StartupStage::Startup, setup_ui)
            .add_system(save_ui)
            .add_system(menu_ui);
    }
}

fn menu_ui(
    mut egui_context: ResMut<EguiContext>,
    mut params: ResMut<UIParams>,
    mut dataset_state: ResMut<DatasetUI>,
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


fn setup_ui(
    mut params: ResMut<UIParams>,
    mut data_param: ResMut<DatasetUI>,
    mut egui_context: ResMut<EguiContext>
) {        
    let root_path: path::PathBuf = params.root_path.clone().into();
    let config_file = root_path.join("ui_config").with_extension("ron");
    // loading configurations of main ui components
    if config_file.exists() {
        eprintln!("loading from config file {}", config_file.to_str().unwrap());
        let config: (String, String) = ron::from_str(&fs::read_to_string(&config_file).unwrap()).unwrap();
        params.load_config(&config.0);

        if let Err(e) = data_param.load_config(&config.1) {
            eprintln!("Unable to load dataset parameters due to {}", e.to_string());
        }
    }

    // startup tasks that one must do to update the ui
    change_font_size(params.font_delta, egui_context.ctx_mut());
}


fn save_ui(
    mut exit: EventReader<AppExit>, 
    mut closed: EventReader<WindowClosed>, 
    mut closed2: EventReader<WindowCloseRequested>,
    params: Res<UIParams>,
    dataset_params: Res<DatasetUI>
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

        eprintln!("saving config");
        let config_file = root_path.join("ui_config").with_extension("ron");

        // getting configurations of main ui components
        let main_ui_config = params.config();
        let data_ui_config = dataset_params.config();
        eprintln!("data ui params {}", data_ui_config);

        let serialized = ron::to_string(&(main_ui_config, data_ui_config)).unwrap();
        fs::write(&config_file, serialized).unwrap();
    }
}


/// Main configuration state for the entire ui
/// as in, the ui can be constructed solely from these parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct UIParams {
    pub root_path: String,
    pub font_delta: f32,
    open_panel: OpenPanel,
}

/// The state for the entire app, which characterizes the two main modes of operation
/// Menu involves only light ui tasks, while Trainer may involve some heavy compute, 
/// (which may run on another thread), and rendering.
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
enum State {
    Menu,
    Trainer
}

/// State for panel opened in the ui
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
enum OpenPanel {
    Models,
    Datasets,
    Misc,
    Train
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
        UIParams { open_panel: OpenPanel::Models, font_delta: 4.0, root_path: "".to_string() }
    }
}

/*
impl<T: Param> Param for Vec<T> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        for (i, transform) in (0..self.len()).zip(self.iter_mut()) {
            ui.label(format!("transform {}", i));
            ui.end_row();
            transform.ui(ui);
        }
    }

    fn config(&self) -> String {
        let temp: Vec<String> = self.iter().map(|x| x.config()).collect();
        ron::to_string(&temp).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let temp: Vec<String> = ron::from_str(config).context("Param Trait: unable to serialize from vec")?;
        for (x, y) in self.iter_mut().zip(temp.iter()) {
            x.load_config(y)?;
        }
        Ok(())
    }
}


impl<K, V> Param for HashMap<K, V> 
where K: Serialize + DeserializeOwned + Hash + Eq + Send + Sync + Display + Clone, V: Param {
    fn ui(&mut self, ui: &mut egui::Ui) {
        for (k, v) in self.iter_mut() {
            ui.label(format!("id {}", k));
            v.ui(ui);
            ui.end_row();
        }
    }

    fn config(&self) -> String {
        let temp: HashMap<K, String>  = self.iter().map(|(k, v)| (k.clone(), v.config())).collect();
        ron::to_string(&temp).unwrap()
    }

    fn load_config(&mut self, config: &str) -> Result<()> {
        let temp: HashMap<K, String> = ron::from_str(config).context("Param Trait: unable to load from HashMap")?;
        for (k, v) in self.iter_mut() {
            if let Some(s) = temp.get(k) {
                v.load_config(s)?;
            }
        }
        Ok(())
    }
}
*/

fn change_font_size(font_delta: f32, ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::new(18.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Monospace, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::new(14.0 + font_delta, egui::FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::new(10.0 + font_delta, egui::FontFamily::Proportional));
    ctx.set_style(style);
}
