use std::collections::{HashMap, VecDeque};

use anyhow::{Result, Error};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use model_lib::models::{Train, TrainProgress, Log};
use super::AppState;
use super::super::model_configs::baseline;
use crate::{Config, UI};

pub trait TrainConfig: Train + Config + UI {}
impl<T: Train + Config + UI> TrainConfig for T {}


#[derive(Serialize, Deserialize, Default)]
struct TrainingUI {
    baseline: ModelEnviron<baseline::BaselineParams>,
    model: Models
}

impl TrainingUI {
    fn ui(&mut self, ui: &mut egui::Ui, logs: &TrainLogs, event: ResMut<State<AppState>>) {

    }
}


pub enum Visuals {
}

pub struct TrainResource {
    train: Option<TrainInstance>,
    console: Console,
    vis: Visuals
}

pub struct Console {
    console_msgs: VecDeque<String>,
    max_console_msgs: usize,
}

impl Console {
    fn new(n_logs: usize) -> Self {
        Console { console_msgs: VecDeque::new(), max_console_msgs: n_logs }
    }

    fn insert_msg(&mut self, msg: String) {
        self.console_msgs.push_back(msg);
        if self.console_msgs.len() > self.max_console_msgs {
            self.console_msgs.pop_front();
        }
    }

    fn ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for text in &self.console_msgs {
                ui.label(text);
            }
        });
    }
}

struct TrainPanePlugin;
impl Plugin for TrainPanePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_system(handle_logs)
            .add_system_set(
                SystemSet::on_enter(AppState::Trainer)
                    .with_system(training_system)   
            );
    }
}

/// this system is run regardless of AppState, as we want to switch between
/// menu and training pane regardless of models running, do this so channels
/// do not grow unbounded in size.
pub fn handle_logs(
    mut train_res: ResMut<TrainResource>, 
    mut logs: ResMut<TrainLogs>
) {
    if train_res.train.is_some() {
        loop { // this is appease the borrow checker
            let event = train_res.train.as_ref().unwrap().progress.recv.try_recv();
            if let Err(_) = event {
                break;
            }
            match event.unwrap() {
                Log::PLOT(name, x, y) => {
                    let msg = format!("plot {name}: x: {x}, y: {y}");
                    train_res.console.insert_msg(msg);
                    
                    // TOOD: add log to logs
                }
            }
        }
    }
}


/// This system corresponds to the egui component of the training pane
/// handling plots, etc.
pub fn training_system(
    mut egui_context: ResMut<EguiContext>,
    mut state: ResMut<State<AppState>>, 
    mut train_res: ResMut<TrainResource>,
    mut logs: ResMut<TrainLogs>) {
    
    let logs = &mut *logs;
    egui::Window::new("train").show(egui_context.ctx_mut(), |ui| {
        // make it so that going back to menu does not suspend current training progress
        if ui.button("back to menu").clicked() {
            state.set(AppState::Menu).unwrap();
        }
        if ui.button("stop training").clicked() {

        }
        if train_res.train.is_some() {

            // the console and log graphs are part of the fore-ground egui panel
            // while any background rendering stuff is happening in a separate system, taking TrainResource as a parameter
            ui.collapsing("console", |ui| {
                train_res.console.ui(ui);
            });

            // TODO: Add plotting utilites
        } else {
            ui.label("no models selected");
        }

    });
}




#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone)]
pub enum Models {
    BASELINE
}

impl std::fmt::Display for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Models::BASELINE => write!(f, "baseline")
        }
    }
}

impl Default for Models {
    fn default() -> Self {
        Models::BASELINE
    }
}

/// Environment responsible for tracking past configs, and producing a new training instance
/// from a config snapshot
/// TODO: make this store checkpoints as well, so that models can be 'restored' from a checkpoint
#[derive(Serialize, Deserialize, Default)]
pub struct ModelEnviron<Config> {
    name: String,
    config: Config,
    runs: Vec<Config>,
    version_num: u32
}

impl<C: TrainConfig + Default> ModelEnviron<C> {
    fn new(name: String) -> Self {
        Self { name: name.to_string(), config: C::default(), runs: Vec::new(), version_num: 0 }
    }

    fn start_training(&mut self) -> TrainInstance {
        TrainInstance { name: self.name.clone(), progress: self.config.build() }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("config for {}", self.name));
        self.config.ui(ui);
        ui.separator();
        ui.vertical(|ui| {
            ui.collapsing("past configs", |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, c) in &mut self.runs.iter_mut().enumerate() {
                        ui.label(format!("config {}", i));         
                        c.ui(ui);
                    }
                });
            });
        });
    }

}

/// The struct to pass to the training phase, which contains the senders and receivers
/// for the various rendering stages
pub struct TrainInstance {
    name: String,
    progress: TrainProgress,
}



/// Struct containing all the logs associated with every run
/// configs gets 'collapsed' into a common dictionary representation for displaying purposes
#[derive(Default, Serialize, Deserialize)]
pub struct TrainLogs {
    models: HashMap<String, ModelLogInfo>
}

/// Every model gets is own ModelLogInfo struct, which tracks every log of that model
#[derive(Serialize, Deserialize)]
pub struct ModelLogInfo {
    name: String,  // name of the class of models
    runs: Vec<RunInfo>,   
}

/// This struct represents an individual training run
#[derive(Serialize, Deserialize)]
pub struct RunInfo {
    pub version: u32,  // id for this run
    pub name: String,  // specific name for this run
    pub dataset: String,
    pub plots: HashMap<String, Vec<(f32, f32)>>,
    pub config: HashMap<String, String>
}


impl TrainLogs {
    fn new_model(&mut self, name: String) -> Result<()> {
        if self.models.contains_key(&name) {
            return Err(Error::msg(format!("already contains model {name}")));
        }
        self.models.insert(name.clone(), ModelLogInfo { name: name.to_string(), runs: Vec::new() });
        Ok(())
    }

    fn insert_run(&mut self, model: String, run: RunInfo) -> Result<()> {
        if let Some(x) = self.models.get_mut(&model) {
            x.runs.push(run);
        } else {
            return Err(Error::msg(format!("Does not contain model {}", model)));
        }
        Ok(())
    }
}

#[test]
fn serialize_test() {
    #[derive(Default, Serialize, Deserialize)]
    struct TestParams {
        a: usize,
        b: String,
        c: f32
    }

    let a = TestParams::default();
    let json = ron::to_string(&a).unwrap();
    println!("{}", json);
    let out: HashMap<String, String> = ron::from_str(&json).expect("unable to convert to hashmap");
    println!("{:?}", out);
}