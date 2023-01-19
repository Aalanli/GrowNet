use std::collections::{HashMap, VecDeque};

use anyhow::{Result, Error};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use model_lib::models::{Train, TrainProcess, TrainCommand, Log};
use super::AppState;
use super::super::model_configs::baseline;
use crate::{Config, UI};

pub trait TrainConfig: Train + Config + UI {}
impl<T: Train + Config + UI + Clone> TrainConfig for T {}


#[derive(Serialize, Deserialize, Default)]
pub struct TrainingUI {
    baseline: ModelEnviron<baseline::BaselineParams>,
    baseline_ver: u32,
    model: Models
}

impl TrainingUI {
    pub fn ui(&mut self, ui: &mut egui::Ui, logs: &TrainLogs) -> Option<TrainInstance> {
        ui.horizontal(|ui| {
            // the left most panel showing a list of model options
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.model, Models::BASELINE, "baseline");
            });

            ui.vertical(|ui| {
                match self.model {
                    Models::BASELINE => {
                        self.baseline.ui(ui);
                    }
                }
            });
        });

        // TODO: add some keybindings to certain buttons
        if ui.button("start training").clicked() {
            match self.model {
                Models::BASELINE => {
                    Some(
                        TrainInstance { run: 
                            RunInfo {
                                model_class: "BASELINE".to_string(),
                                version: self.baseline_ver,
                                dataset: "CIFAR10".to_string(),
                                ..default()
                            }, run_starter: Box::new(self.baseline.config.clone()) }
                    )
                }
            }
        } else {
            None
        }
    }
}


pub enum Visuals {
}


pub struct TrainResource {
    run_queue: VecDeque<TrainInstance>,
    cur_instance: Option<TrainInstance>,
    train_process: Option<TrainProcess>,
    console: Console,
    vis: Visuals
}

impl TrainResource {
    /// schedules a new run on the end of the run queue
    pub fn add_run(&mut self, run: TrainInstance) {
        self.run_queue.push_back(run);
    }

    /// removes all runs scheduled, does not kill current active run, if any
    pub fn clear_runs(&mut self) {
        self.run_queue.clear();
    }

    /// removes all runs scheduled, and kills current active run if any
    /// blocks until the current active process (if any) is killed
    pub fn clean_runs(&mut self) -> Result<()> {
        self.clear_runs();
        if let Some(t) = &mut self.train_process {
            t.send.send(TrainCommand::KILL)?;
            loop {
                if let Ok(log) = t.recv.try_recv() {
                    match log {
                        Log::KILLED => {
                            return Ok(());
                        },
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
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


/// This system is run regardless of AppState, as we want to switch between
/// menu and training pane regardless of models running,
/// Consequently, this system handles logging from any active training processes.
/// scheduling new processes, and killing old ones.
/// 
/// This is setup so eventually, we can have hyperparmeter sweeps with relative ease,
/// all the information for this is right here in this system, for max worker threads,
/// gpu scheduling, etc.
pub fn handle_training(
    mut train_res: ResMut<TrainResource>, 
    mut logs: ResMut<TrainLogs>,
) {
    let TrainResource { 
        run_queue, 
        cur_instance,
        train_process, 
        console, 
        vis } = &mut *train_res;
    
    
    // logging stuff
    if let Some(process) = train_process {
        while let Ok(log) = process.recv.try_recv() {

            match &log {
                Log::PLOT(name, x, y) => {
                    let msg = format!("plot {name}: x: {x}, y: {y}");
                    console.insert_msg(msg);
                    
                    // TOOD: add log to logs
                }
                Log::KILLED => {
                    let training_name = cur_instance.as_ref().unwrap().run.run_name();
                    console.insert_msg(format!("killed successfully {}", training_name));
                }
            }

            cur_instance.as_mut().unwrap().run.log(log);
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
        if train_res.train_process.is_some() {

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
    pub_runs: Vec<Config>, // dumb method, since we need ui method to display, but can also mutate this, so we keep backup
    checked: isize, // each vec has a checkbox, this is the index which is checked
    version_num: u32
}

impl<C: TrainConfig + Default + Clone> ModelEnviron<C> {
    fn new(name: String) -> Self {
        Self { name: name.to_string(), checked: -1, ..default() }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("config for {}", self.name));
        self.config.ui(ui);
        ui.separator();
        // reset config
        if self.checked < 0 { // nothing is checked, reset to default
            if ui.button("reset config").clicked() {
                self.config = C::default();
            }
        } else { // something is checked, default to past config
            if ui.button(format!("reset config with checked option {}", self.checked)).clicked() {
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

/// The struct to pass to the training phase, which contains the senders and receivers
/// for the various rendering stages
pub struct TrainInstance {
    run: RunInfo,
    run_starter: Box<dyn TrainConfig>,
}

/// Struct containing all the logs associated with every run
/// configs gets 'collapsed' into a common dictionary representation for displaying purposes
#[derive(Default, Serialize, Deserialize)]
pub struct TrainLogs {
    models: Vec<RunInfo> // flattened representation, small enough number of runs to justify not using hashmap
}

/// This struct represents an individual training run
#[derive(Serialize, Deserialize, Default)]
pub struct RunInfo {
    pub model_class: String, // name for the class of models this falls under
    pub version: u32,        // id for this run
    pub comments: String,
    pub dataset: String,
    pub plots: HashMap<String, Vec<(f32, f32)>>,
    pub config: Option<HashMap<String, String>>  // TODO: convert types to this more easily
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }
    
    pub fn log(&mut self, log: Log) {

    }

    pub fn reset(&mut self) {}
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