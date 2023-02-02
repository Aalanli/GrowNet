use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::super::model_configs::baseline;
use super::{AppState, UIParams};
use crate::{Config, UI};
use model_lib::models::{self, Models};



#[derive(Resource, Deref, DerefMut, Default, Serialize, Deserialize)]
pub struct TrainData(models::TrainData);


#[derive(Resource, Deref, DerefMut, Default, Serialize, Deserialize)]
pub struct TrainEnviron(models::TrainEnviron);



pub struct TrainUIPlugin;
impl Plugin for TrainUIPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<TrainError>()
            .insert_resource(TrainData::default())
            .insert_resource(TrainEnviron::default())
            .insert_resource(TrainingUI::default())
            .add_startup_system(setup_train_ui)
            .add_system_set(SystemSet::on_update(AppState::Close).with_system(save_train_ui));
    }
}

/// possibly load any training state from disk
fn setup_train_ui(
    params: Res<UIParams>,
    mut train_data: ResMut<TrainData>,
    mut train_environ: ResMut<TrainEnviron>,
    mut train_ui: ResMut<TrainingUI>,
) {
    // load config files if any
    fn try_deserialize<T: DeserializeOwned>(x: &mut T, path: &std::path::PathBuf) {
        if path.exists() {
            eprintln!("deserializing {}", path.to_str().unwrap());
            let reader = std::fs::File::open(path).expect("unable to open file");
            match bincode::deserialize_from(reader) {
                Ok(y) => { *x = y; },
                Err(e) => { eprintln!("unable to deserialize\n{}", e); }
            }
        } else {
            eprintln!("{} does not exist", path.to_str().unwrap());
        }
    }
    let root_path: std::path::PathBuf = params.root_path.clone().into();
    try_deserialize(&mut *train_data, &root_path.join("train_ui_data").with_extension("config"));
    try_deserialize(&mut *train_environ, &root_path.join("train_ui_environ").with_extension("config"));
    try_deserialize(&mut *train_ui, &root_path.join("train_ui").with_extension("config"));
}

/// write train state to disk
fn save_train_ui(
    params: Res<UIParams>,
    train_data: Res<TrainData>,
    train_environ: Res<TrainEnviron>,
    train_ui: Res<TrainingUI>,
) {
    // load configurations from disk
    let root_path: std::path::PathBuf = params.root_path.clone().into();
    
    eprintln!("serializing train_ui");
    // save config files to disk
    let config_file = root_path.join("train_ui_data").with_extension("config");
    let train_data_writer = std::fs::File::create(config_file).unwrap();
    bincode::serialize_into(train_data_writer, &*train_data).expect("unable to serialize train_data");

    let config_file = root_path.join("train_ui_environ").with_extension("config");
    let train_data_writer = std::fs::File::create(config_file).unwrap();
    bincode::serialize_into(train_data_writer, &*train_environ).expect("unable to serialize train_environ");

    let config_file = root_path.join("train_ui").with_extension("config");
    let train_data_writer = std::fs::File::create(config_file).unwrap();
    bincode::serialize_into(train_data_writer, &*train_ui).expect("unable to serialize train_ui");
}


fn console_ui(console: &models::Console, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        for text in &console.console_msgs {
            ui.label(text);
        }
    });
}

fn plot_ui() { todo!() }

/// A fatal training error has occurred
struct TrainError(String);

#[derive(Serialize, Deserialize, Default, Resource)]
pub struct TrainingUI {
    baseline: ConfigEnviron<baseline::BaselineParams>,
    baseline_ver: u32,
    model: Models,
}

impl TrainingUI {
    pub fn ui(&mut self, ui: &mut egui::Ui, train_env: &mut models::TrainEnviron, app_state: &mut State<AppState>) {
        // ui.horizontal squashes the vertical height for some reason
        let space = ui.available_size();
        ui.horizontal(|ui| {
            // to this to prevent that
            ui.allocate_ui(space, |ui| {
                // the left most panel showing a list of model options
                ui.vertical(|ui| {
                    ui.selectable_value(&mut self.model, Models::BASELINE, "baseline");

                    // TODO: add some keybindings to certain buttons
                    if ui.button("start training").clicked() {
                        match self.model {
                            // TODO: check if current model is running
                            Models::BASELINE => {
                                let config = self.baseline.get_config();
                                train_env.baseline.set_config(config);
                                app_state.set(AppState::Trainer).unwrap();
                            }
                        }
                    }
                });
    
                ui.vertical(|ui| match self.model {
                    Models::BASELINE => {
                        self.baseline.ui(ui);
                    }
                });
            });
        });

    }
}

fn handle_logs(
    mut train_env: ResMut<TrainEnviron>,
    mut train_data: ResMut<TrainData>,
    mut err_ev: EventWriter<TrainError>,
) {
    if train_env.is_running() {
        match train_env.selected() {
            Models::BASELINE => {
                let logs = train_env.baseline.run_data();
                if let Err(e) = train_data.handle_baseline_logs(&logs) {
                    err_ev.send(TrainError(format!("{e}")));
                }
            }
        }
    }
}

/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut train_env: ResMut<TrainEnviron>,
    mut train_data: ResMut<TrainData>,
    mut err_ev: EventWriter<TrainError>,
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        if ui.button("back to menu").clicked() {
            state.set(AppState::Menu).unwrap();
        }
        if ui.button("stop training").clicked() {
            match train_env.selected() {
                Models::BASELINE => {
                    let mut log_buf = Vec::new();
                    train_env.baseline.kill_blocking(&mut log_buf).expect("unable to kill task");
                    // TODO: not so sure about the error handling
                    if let Err(e) = train_data.handle_baseline_logs(&log_buf) {
                        err_ev.send(TrainError(format!("{e}")));
                    }
                }
            }
        }
        if train_env.is_running() {
            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                console_ui(&train_data.console, ui);
            });
            
            // TODO: Add plotting utilites
        } else {
            state.set(AppState::Menu).unwrap();
        }
    });
}


/// Environment responsible for manipulating various configs, and passing them to TrainEnviron to train,
/// this does not know any low-level details about the configs.
#[derive(Serialize, Deserialize, Default)]
pub struct ConfigEnviron<Config> {
    name: String,
    config: Config,
    runs: Vec<Config>,
    pub_runs: Vec<Config>, // dumb method, since we need ui method to display, but can also mutate this, so we keep backup
    checked: Option<usize>,        // each vec has a checkbox, this is the index which is checked
    version_num: u32,
}

impl<C: UI + Config + Default + Clone> ConfigEnviron<C> {
    pub fn new(name: String) -> Self {
        Self {
            name: name.to_string(),
            checked: None,
            ..default()
        }
    }

    pub fn get_config(&self) -> C {
        self.config.clone()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let space = ui.available_size();
        ui.horizontal(|ui| {
            ui.allocate_ui(space, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        // reset current config logic
                        if self.checked.is_none() {
                            // nothing is checked, reset to default
                            if ui.button("reset config").clicked() {
                                self.config = C::default();
                            }
                        } else {
                            // something is checked, default to past config
                            let checked = self.checked.as_mut().unwrap();
                            if ui
                                .button(format!("reset config with past config {}", checked))
                                .clicked()
                            {
                                if let Some(a) = self.runs.get(*checked) {
                                    self.config = a.clone()
                                }
                            }
                        }
                        // save current config logic
                        if ui.button("save config").clicked() {
                            self.pub_runs.push(self.config.clone());
                            self.runs.push(self.config.clone());
                        }
                    });
                    egui::ScrollArea::vertical().id_source("configs").show(ui, |ui| {
                        self.config.ui(ui);
                    });
                });
                
                // implement adding and deletion from config stack
                ui.vertical(|ui| {
                    // TODO: Add checkpoints
                    egui::ScrollArea::vertical().id_source("past configs").show(ui, |ui| {
                        // use pub_runs as dummy display
                        let mut i = 0;
                        while i < self.pub_runs.len() {
                            // allow checked to be negative so it becomes possible for no
                            // option to be checked
                            let mut cur_check = self.checked.is_some() && i == self.checked.unwrap();
                            let mut removed_run = false;
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut cur_check, format!("config {}", i));
                                if ui.button("delete config").clicked() {
                                    self.pub_runs.remove(i);
                                    self.runs.remove(i);
                                    removed_run = true;
                                    if cur_check {
                                        self.checked = None;
                                    }
                                }
                            });
                            if removed_run {
                                continue;
                            }
                            ui.push_id(format!("config panel checked {}", i), |ui| {
                                self.pub_runs[i].ui(ui);
                            });
                            // we don't want past configs to change, so we have an immutable copy
                            self.pub_runs[i] = self.runs[i].clone();
                            // only one option can be checked at a time
                            let checked = self.checked.is_some() && i == self.checked.unwrap();
                            if cur_check {
                                self.checked = Some(i);
                            } else if checked {
                                self.checked = None;
                            }
                            i += 1;
                        }
                    });
                    
                }); 
            });
        });
    }
}