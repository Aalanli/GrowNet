use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::super::model_configs::baseline;
use super::{AppState, UIParams};
use crate::{Config, UI};
use model_lib::models::{self, TrainRecv, TrainSend, TrainProcess, Models};

pub use model_lib::models::{RunInfo};




fn console_ui(console: &models::Console, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        for text in &console.console_msgs {
            ui.label(text);
        }
    });
}

fn plot_ui() { todo!() }


#[derive(Serialize, Deserialize, Default)]
pub struct TrainingUI {
    baseline: ConfigEnviron<baseline::BaselineParams>,
    baseline_ver: u32,
    model: Models,
}

impl TrainingUI {
    pub fn ui(&mut self, ui: &mut egui::Ui, train_env: &mut models::TrainEnviron, app_state: &mut State<AppState>) {
        ui.horizontal(|ui| {
            // the left most panel showing a list of model options
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.model, Models::BASELINE, "baseline");
            });

            ui.vertical(|ui| match self.model {
                Models::BASELINE => {
                    self.baseline.ui(ui);
                }
            });
        });

        // TODO: add some keybindings to certain buttons
        if ui.button("start training").clicked() {
            match self.model {
                // TODO: check if current model is running
                Models::BASELINE => {
                    let config = self.baseline.get_config();
                    train_env.baseline.set_config(config);
                    
                }
            }
        }
    }
}

pub fn handle_logs(
    mut train_env: ResMut<models::TrainEnviron>,
    mut train_data: ResMut<models::TrainData>,
) {
    if train_env.is_running() {
        match train_env.selected() {
            Models::BASELINE => {
                let logs = train_env.baseline.run_data();
                train_data.handle_baseline_logs(&logs);
            }
        }
    }
}

/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
pub fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut train_env: ResMut<models::TrainEnviron>,
    mut train_data: ResMut<models::TrainData>,
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        if ui.button("back to menu").clicked() {
            state.set(AppState::Menu).unwrap();
        }
        if ui.button("stop training").clicked() {
            match train_env.selected() {
                Models::BASELINE => {
                    let logs = train_env.baseline.kill_blocking().expect("unable to kill task");
                    train_data.handle_baseline_logs(&logs);
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
    checked: isize,        // each vec has a checkbox, this is the index which is checked
    version_num: u32,
}

impl<C: UI + Config + Default + Clone> ConfigEnviron<C> {
    pub fn new(name: String) -> Self {
        Self {
            name: name.to_string(),
            checked: -1,
            ..default()
        }
    }

    pub fn get_config(&self) -> C {
        self.config.clone()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("config for {}", self.name));
        self.config.ui(ui);
        ui.separator();
        // reset config
        if self.checked < 0 {
            // nothing is checked, reset to default
            if ui.button("reset config").clicked() {
                self.config = C::default();
            }
        } else {
            // something is checked, default to past config
            if ui
                .button(format!("reset config with checked option {}", self.checked))
                .clicked()
            {
                if let Some(a) = self.runs.get(self.checked as usize) {
                    self.config = a.clone()
                }
            }
        }

        // implement adding and deletion from config stack
        ui.vertical(|ui| {
            ui.collapsing("past configs", |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // use pub_runs as dummy display
                    for (i, c) in &mut self.pub_runs.iter_mut().enumerate() {
                        // allow checked to be negative so it becomes possible for no
                        // option to be checked
                        let mut checked = i as isize == self.checked;
                        ui.checkbox(&mut checked, format!("config {}", i));
                        c.ui(ui);
                        // we don't want past configs to change, so we have an immutable copy
                        *c = self.runs[i].clone();
                        // only one option can be checked at a time
                        if checked {
                            self.checked = i as isize;
                        } else {
                            self.checked = -1;
                        }
                    }
                });
            });
        });
    }
}