use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use bevy::ecs::schedule::ShouldRun;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{AppState, OpenPanel, UIParams, handle_pane_options};
use crate::{Configure, UI};
use model_lib::models::{self};
use model_lib::Config;
use crate::model_configs::immutable_show;

mod run_data;
mod train_systems;

use run_data as run;


pub struct TrainUIPlugin;
impl Plugin for TrainUIPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<TrainError>()
            .add_event::<KillTraining>()
            .insert_resource(run::ModelPlots::default())
            .insert_resource(run::Console::default())
            .insert_resource(TrainingUI::default())
            .add_startup_system(setup_train_ui)
            .add_system_set(
                SystemSet::on_update(AppState::Models)
                    .label("train_menu")
                    .with_system(training_ui_system)
            )
            .add_system(run_baseline)
            .add_system_set(SystemSet::on_update(AppState::Close).with_system(save_train_ui));
    }
}

/// possibly load any training state from disk
fn setup_train_ui(
    params: Res<UIParams>,
    mut plots: ResMut<run::ModelPlots>,
    mut console: ResMut<run::Console>,
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
    try_deserialize(&mut *plots, &root_path.join("model_plots").with_extension("config"));
    try_deserialize(&mut *console, &root_path.join("model_console").with_extension("config"));
    try_deserialize(&mut *train_ui, &root_path.join("train_ui").with_extension("config"));
}

/// write train state to disk
fn save_train_ui(
    params: Res<UIParams>,
    plots: Res<run::ModelPlots>,
    console: Res<run::Console>,
    train_ui: Res<TrainingUI>,
) {
    // load configurations from disk
    let root_path: std::path::PathBuf = params.root_path.clone().into();

    fn serialize<T: Serialize>(x: &T, path: &std::path::PathBuf) {
        eprintln!("serializing {}", path.to_str().unwrap());
        let train_data_writer = std::fs::File::create(path).unwrap();
        bincode::serialize_into(train_data_writer, x).expect("unable to serialize");
    }

    eprintln!("serializing train_ui");
    // save config files to disk
    serialize(&*plots, &root_path.join("model_plots").with_extension("config"));
    serialize(&*console, &root_path.join("model_console").with_extension("config"));
    serialize(&*train_ui, &root_path.join("train_ui").with_extension("config"));

}

/// A fatal training error has occurred
#[derive(Deref, DerefMut)]
struct TrainError(String);

struct KillTraining;

#[derive(Serialize, Deserialize, Resource)]
pub struct TrainingUI {
    baseline: ConfigEnviron,
    model: run::Models,
}

impl Default for TrainingUI {
    fn default() -> Self {
        Self { 
            baseline: ConfigEnviron::new("baseline",models::baseline::baseline_config()), 
            model: run::Models::BASELINE 
        }
    }
}

fn training_ui_system(
    mut commands: Commands,
    mut egui_context: ResMut<EguiContext>,
    mut app_state: ResMut<State<AppState>>,
    mut params: ResMut<UIParams>,
    mut train_ui: ResMut<TrainingUI>,
    mut local_spawn_error: Local<Option<String>>,
) {
    egui::CentralPanel::default().show(egui_context.ctx_mut(), |ui| {
        handle_pane_options(ui, &mut params.open_panel);

        if std::mem::discriminant(&params.open_panel) != std::mem::discriminant(&OpenPanel::Models) {
            app_state.set(AppState::Menu).unwrap(); // should be fine to not return here
        }

        let space = ui.available_size();
        ui.horizontal(|ui| {
            // to this to prevent that
            ui.allocate_ui(space, |ui| {
                // the left most panel showing a list of model options
                ui.vertical(|ui| {
                    ui.selectable_value(&mut train_ui.model, run::Models::BASELINE, "baseline");

                    if let Some(err) = &*local_spawn_error {
                        ui.label(egui::RichText::new(err).color(egui::Color32::DARK_RED));
                    }

                    // TODO: add some keybindings to certain buttons
                    // entry point for launching training
                    if ui.button("start training").clicked() {
                        match train_ui.model {
                            run::Models::BASELINE => {
                                spawn_baseline(&train_ui.baseline.config, train_ui.baseline.version_num as usize)
                                    .map_or_else(|e| {
                                        *local_spawn_error = Some(e.to_string());
                                    }, |bundle| {
                                        commands.spawn(bundle);
                                        train_ui.baseline.version_num += 1;
                                        app_state.set(AppState::Trainer).unwrap();
                                    });
                            }
                        }
                    }
                });

                // update any configurations using the ui
                ui.vertical(|ui| match train_ui.model {
                    run::Models::BASELINE => {
                        train_ui.baseline.ui(ui);
                    }
                });
            });
        });

    });
}

/// This struct represents an individual training run, it has the information to restart itself
#[derive(Serialize, Deserialize, Default, Clone, Component)]
pub struct RunInfo {
    pub config: Config,             // TODO: convert types to this more easily,
    pub model_class: String, // name for the class of models this falls under
    pub version: usize,      // id for this run
    pub comments: String,
    pub dataset: String,
    pub checkpoints: Vec<(f32, std::path::PathBuf)>, // (step, path)
    pub err_status: Option<String>, // True is returned successfully, false if Killed mid-run
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }

    pub fn show_basic(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            egui::ScrollArea::vertical().id_source("past runs").show(ui, |ui| {

                ui.collapsing(self.run_name(), |ui| {
                    ui.collapsing("comments", |ui| {
                        ui.label(&self.comments);
                    });
                    ui.collapsing("checkpoints", |ui| {
                        egui::ScrollArea::vertical().id_source("click checkpoints").show(ui, |ui| {

                            for (j, checkpoint) in self.checkpoints.iter() {
                                // TODO: when checkpoint is clicked, show loss as well
                                ui.horizontal(|ui| {
                                    ui.label(format!("step {}", j));
                                    ui.label(checkpoint.to_str().unwrap());
                                });
                            }
                        });
                    });
                    ui.label(format!("error status: {:?}", self.err_status));
                    ui.label(format!("dataset: {}", self.dataset));
                    ui.label(format!("model class: {}", self.model_class));
                    ui.collapsing("run configs", |ui| {
                        immutable_show(&self.config, ui);
                    });
                });

            });
        });
    }
}

#[derive(Deref, DerefMut, Component)]
pub struct ConfigComponent(Config);

/// Environment responsible for manipulating various configs, and passing them to TrainEnviron to train,
/// this does not know any low-level details about the configs.
#[derive(Serialize, Deserialize)]
pub struct ConfigEnviron {
    name: String,
    config: Config,
    default: Config,
    saved_configs: CheckedList<Config>,

    saved_runs: Vec<RunInfo>,
    pub_runs: Vec<RunInfo>,

    checked_runinfo: usize,
    checked_checkpoint: Vec<usize>,
    version_num: u32,
}

impl ConfigEnviron {
    pub fn new(name: &str, config: Config) -> Self {
        Self {
            name: name.to_string(),
            config: config.clone(),
            default: config,
            saved_configs: CheckedList { header: name.to_string() + " saved configs", deletion: true, ..default() },
            saved_runs: Vec::new(),
            pub_runs: Vec::new(),
            checked_runinfo: 0,
            version_num: 0,
            checked_checkpoint: Vec::new(),
        }
    }

    pub fn get_config(&self) -> Config {
        self.config.clone()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let space = ui.available_size();
        ui.horizontal(|ui| {
            ui.allocate_ui(space, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        // reset current config logic
                        if self.saved_configs.checked.is_none() {
                            // nothing is checked, reset to default
                            if ui.button("reset config").clicked() {
                                self.config.update(&self.default).unwrap();
                            }
                        } else {
                            // something is checked, default to past config
                            let checked = self.saved_configs.checked.unwrap();
                            if ui
                                .button(format!("reset config with past config {}", checked))
                                .clicked()
                            {
                                if let Some(a) = self.saved_configs.get_checked() {
                                    self.config.update(a).unwrap();
                                }
                            }
                        }
                        // save current config logic
                        if ui.button("save config").clicked() {
                            self.saved_configs.push(self.config.clone());
                        }
                    });
                    egui::ScrollArea::vertical().id_source("configs").show(ui, |ui| {
                        self.config.ui(ui);
                    });
                });
                
                // implement adding and deletion from config stack
                self.saved_configs.ui(ui, |x, y| { x.update(&y).unwrap(); });

                // show past training runs
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical().id_source("past runs").show(ui, |ui| {
                        for (i, run) in self.pub_runs.iter_mut().enumerate() {
                            let mut checked = i == self.checked_runinfo;
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut checked, format!("run {i}"));
                                // todo
                                if ui.button("re-run").clicked() {

                                }

                                if ui.button("new-run").clicked() {

                                }

                                if ui.button("set config").clicked() {

                                }
                            });
                            ui.collapsing(run.run_name(), |ui| {
                                ui.collapsing("comments", |ui| {
                                    ui.label(&run.comments);
                                });
                                ui.collapsing("checkpoints", |ui| {
                                    egui::ScrollArea::vertical().id_source("click checkpoints").show(ui, |ui| {
                                        let mut checked = self.checked_checkpoint.get(self.checked_runinfo)
                                            .map_or(run.checkpoints.len(), |x| *x);
                                        for (j, checkpoint) in run.checkpoints.iter().enumerate() {
                                            // TODO: when checkpoint is clicked, show loss as well
                                            let mut check = checked == j;
                                            ui.checkbox(&mut check, "");
                                            ui.horizontal(|ui| {
                                                ui.label(format!("step {}", checkpoint.0));
                                                ui.label(checkpoint.1.to_str().unwrap());
                                            });
                                            if check {
                                                checked = j;
                                            }
                                        }

                                        self.checked_checkpoint.get_mut(self.checked_runinfo)
                                            .map(|x| { *x = checked; });
                                    });
                                });
                                ui.label(format!("error status: {:?}", run.err_status));
                                ui.label(format!("dataset: {}", run.dataset));
                                ui.label(format!("model class: {}", run.model_class));
                                ui.collapsing("run configs", |ui| {
                                    run.config.ui(ui);
                                });
                            });

                            if checked {
                                self.checked_runinfo = i;
                            }
                        }
                    });
                });
            });
        });
    }
}

#[derive(Default, Serialize, Deserialize)]
struct CheckedList<T> {
    header: String,
    pub_ui: Vec<T>,
    saved: Vec<T>,
    item_open: Vec<bool>,
    open: bool,
    deletion: bool,
    checked: Option<usize>,
}

impl<T: UI + Clone> CheckedList<T> {
    fn get_checked(&self) -> Option<&T> {
        self.checked.map(|x| &self.saved[x])
    }

    fn push(&mut self, x: T) {
        self.item_open.push(false);
        self.saved.push(x.clone());
        self.pub_ui.push(x);
    }

    fn close_all(&mut self) {
        self.item_open.iter_mut().for_each(|x| *x = false);
    }

    fn ui(&mut self, ui: &mut egui::Ui, update_fn: impl Fn(&mut T, &T)) {
        let is_open = egui::CollapsingHeader::new(&self.header).default_open(self.open).show(ui, |ui| {
            egui::ScrollArea::vertical().id_source("past configs").show(ui, |ui| {
                // use pub_runs as dummy display
                let mut i = 0;
                while i < self.pub_ui.len() {
                    // allow checked to be negative so it becomes possible for no
                    // option to be checked
                    let mut cur_check = self.checked.is_some() && i == self.checked.unwrap();
                    let mut removed_run = false;
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut cur_check, format!("config {}", i));
                        if self.deletion && ui.button("delete config").clicked() {
                            self.pub_ui.remove(i);
                            self.saved.remove(i);
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
                        let is_open = egui::CollapsingHeader::new("").default_open(self.item_open[i]).show(ui, |ui| {
                            self.pub_ui[i].ui(ui);
                        }); 
                        self.item_open[i] = is_open.fully_open();
                    });

                    // we don't want past configs to change, so we have an immutable copy
                    update_fn(&mut self.pub_ui[i], &self.saved[i]);
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
        self.open = is_open.fully_open();
    }
}


/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>,
    mut local_errors: Local<Vec<String>>, // some local error messages possibly relayed by err_ev
    mut killer: EventWriter<KillTraining>,
    mut errors: EventReader<TrainError>,

    plots: Res<run::ModelPlots>,
    console: Res<run::Console>,

    running: Query<&RunInfo>
) {
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("back to menu").clicked() {
                    state.set(AppState::Menu).unwrap();
                }
                if ui.button("stop training").clicked() {
                    killer.send(KillTraining);
                }
            });

            for msg in errors.iter() {
                local_errors.push((*msg).clone());
            }

            if local_errors.len() > 0 {
                for i in &*local_errors {
                    ui.label(i);
                }
            }

            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                console.console_ui(ui);
            });

            for run in running.iter() {
                run.show_basic(ui);
            }

            // TODO: Add plotting utilites

        });
    });
}

#[derive(Component, Deref, DerefMut)]
struct BaseTrainProcess(run::models::TrainProcess);

fn spawn_baseline(config: &Config, ver: usize) -> Result<(BaseTrainProcess, RunInfo)> {
    Ok((
        BaseTrainProcess(run::models::baseline::build(config)?),
        RunInfo {
            model_class: "baseline".into(),
            version: ver,
            dataset: "cifar10".into(),
            config: config.clone(),
            ..Default::default()
        }
    ))
}


/// Add plots and run data contributions to their respective containers
fn run_baseline(
    mut err_ev: EventWriter<TrainError>,

    mut plots: ResMut<run::ModelPlots>,
    mut console: ResMut<run::Console>,
    mut runs: Query<(&mut RunInfo, &mut BaseTrainProcess)>,
) {
    for (mut info, mut train_proc) in runs.iter_mut() {
        if train_proc.is_running() {
            let msgs = train_proc.try_recv();
            for msg in msgs {

            }
        }
    }
}

// fn convert_recv(

//     recv: models::TrainRecv,
//     logs: &mut Vec<Log>,
// ) {
//     match recv {
//         models::TrainRecv::KILLED => {
//             logs.push(Log::CONSOLE(format!("{} finished training", &self.run_info.as_ref().unwrap().run_name())));
//             let mut run_info = std::mem::replace(&mut self.run_info, None).unwrap();
//             run_info.err_status = None;
//             logs.push(Log::RUNINFO(run_info));
//             self.run = None;
//         }
//         models::TrainRecv::PLOT(name, x, y) => {
//             logs.push(Log::CONSOLE(format!("Logged plot name: {}, x: {}, y: {}", &name, x, y)));
//             logs.push(Log::PLOT(name, self.run_name().unwrap(), x, y));
//         }
//         models::TrainRecv::FAILED(error_msg) => {
//             logs.push(Log::CONSOLE(format!("Err {} while training {}", error_msg, &self.run_info.as_ref().unwrap().run_name())));
//             let mut run_info = std::mem::replace(&mut self.run_info, None).unwrap();
//             run_info.err_status = Some(error_msg.clone());
//             logs.push(Log::RUNINFO(run_info));
//             self.run = None;
//             logs.push(Log::ERROR(error_msg));
//         }
//         models::TrainRecv::CHECKPOINT(stepno, path) => {
//             if self.run_info.is_some() {
//                 self.run_info.as_mut().unwrap().checkpoints.push((stepno, path));
//             }
//         }
//     }

// }
