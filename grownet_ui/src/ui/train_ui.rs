use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::super::model_configs::baseline;
use super::{AppState, UIParams};
use crate::{Config, UI};
use model_lib::models::{self, Log, TrainCommand, TrainProcess, Models};

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
    baseline: ModelEnviron<baseline::BaselineParams>,
    baseline_ver: u32,
    model: Models,
}

impl TrainingUI {
    pub fn ui(&mut self, ui: &mut egui::Ui, logs: &models::TrainEnviron) {
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
                Models::BASELINE => {}
            }
        }
    }
}


/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
pub fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut train_res: ResMut<models::TrainEnviron>,
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        if ui.button("back to menu").clicked() {
            state.set(AppState::Menu).unwrap();
        }
        if ui.button("stop training").clicked() {
            // TODO: handle error
            let _e = train_res.kill();
        }
        if train_res.is_running() {
            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                console_ui(&train_res.console, ui);
            });

            // TODO: Add plotting utilites
        } else {
            ui.label("no models selected");
        }
    });
}

/*

/// This system is run regardless of AppState, as we want to switch between
/// menu and training pane regardless of models running,
/// Consequently, this system handles logging from any active training processes.
/// scheduling new processes, and killing old ones.
///
/// This is setup so eventually, we can have hyperparmeter sweeps with relative ease,
/// all the information for this is right here in this system, for max worker threads,
/// gpu scheduling, etc.
pub fn handle_logging(
    mut run_queue: ResMut<RunQueue>,
    mut train_res: ResMut<TrainResource>,
    mut logs: ResMut<ModelLogs>,
    params: Res<UIParams>,
    mut stop_event: EventReader<StopTraining>,
) {
    let TrainResource {
        cur_instance,
        train_process,
        console,
        ..
    } = &mut *train_res;

    // A STOPTraining event is sent
    for _ in stop_event.iter() {
        if train_process.is_some() {
            // if a process is running, kill it
            // TODO: handle this error better
            train_process
                .as_mut()
                .unwrap()
                .send
                .send(TrainCommand::KILL)
                .expect("unable to send kill command");
        } else {
            // else remove front facing tasks in the queue
            run_queue.pop_front();
        }
    }

    // this part is for spawning processes if there isn't one already spawned
    if train_process.is_none() {
        match params.train_schedule {
            TrainProcessSchedule::ONE => {
                // only one process allowed in the queue at a time,
                // if more than one, kills the current task and spawns the last from the queue
                if let Some(task) = run_queue.pop_back() {
                    *train_process = Some(task.run_starter.build());
                    *cur_instance = Some(task);
                }
                run_queue.clear();
            }
            TrainProcessSchedule::LINE => {
                // more than one training task is allowed to be in the queue at the same time,
                // they are processed in order, first in first out
                if let Some(task) = run_queue.pop_front() {
                    *train_process = Some(task.run_starter.build());
                    *cur_instance = Some(task);
                }
            }
        }
    }

    if train_process.is_some() {
        while let Ok(log) = train_process.as_mut().unwrap().recv.try_recv() {
            match &log {
                Log::PLOT(name, x, y) => {
                    let msg = format!("plot {name}: x: {x}, y: {y}");
                    console.insert_msg(msg);

                    // TOOD: add log to logs
                }
                Log::KILLED => {
                    let training_name = cur_instance.as_ref().unwrap().run.run_name();
                    console.insert_msg(format!("killed successfully {}", training_name));

                    // after the process has died, we add its logged results to logs
                    // and replace it with None
                    let last_instace = std::mem::replace(cur_instance, None).unwrap();
                    logs.as_mut().models.push(last_instace.run);
                    // safe to free this since the process has already
                    // in the next frame, the if clause above will spawn a new task
                    *train_process = None;
                }
            }

            cur_instance.as_mut().unwrap().run.log(log);
        }
    }
}

*/


/// Environment responsible for tracking past configs, and producing a new training instance
/// from a config snapshot
/// TODO: make this store checkpoints as well, so that models can be 'restored' from a checkpoint
#[derive(Serialize, Deserialize, Default)]
pub struct ModelEnviron<Config> {
    name: String,
    config: Config,
    runs: Vec<Config>,
    pub_runs: Vec<Config>, // dumb method, since we need ui method to display, but can also mutate this, so we keep backup
    checked: isize,        // each vec has a checkbox, this is the index which is checked
    version_num: u32,
}

impl<C: UI + Config + Default + Clone> ModelEnviron<C> {
    fn new(name: String) -> Self {
        Self {
            name: name.to_string(),
            checked: -1,
            ..default()
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
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