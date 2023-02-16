use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use crossbeam::channel::{Sender, Receiver};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
use serde::{Deserialize, Serialize};

pub use model_lib::{models, Config};
pub use models::{TrainProcess, TrainRecv, TrainSend};
pub use crate::ui::OperatingState;

use crate::ops;
use crate::CONFIG_PATH;

/// Plugin to instantiate all run data resources, and saving/loading logic
pub struct RunDataPlugin;
impl Plugin for RunDataPlugin {
    fn build(&self, app: &mut App) {
        let (send, recv) = crossbeam::channel::unbounded();
        let run_sender = RunSend(send);
        let run_recv = RunRecv(recv);
        app
            .add_event::<Despawn>()
            .add_event::<Kill>()
            .insert_resource(run_sender)
            .insert_resource(run_recv)
            .insert_resource(ModelPlots::default())
            .insert_resource(Console::default())
            .insert_resource(RunStats::default())
            .add_startup_system(setup_run_data)
            .add_system_set(
                SystemSet::on_update(OperatingState::Close).with_system(save_run_data));
    }
}

/// possibly load run data from disk
fn setup_run_data(
    mut plots: ResMut<ModelPlots>,
    mut console: ResMut<Console>,
) {
    eprintln!("loading run data");
    ops::try_deserialize(&mut *plots, &(CONFIG_PATH.to_owned() + "/model_plots.config").into());
    ops::try_deserialize(&mut *console, &(CONFIG_PATH.to_owned() + "/model_console.config").into());
}

/// write run data to disk
fn save_run_data(
    plots: Res<ModelPlots>,
    console: Res<Console>,
) {
    // load configurations from disk
    let root_path: std::path::PathBuf = CONFIG_PATH.into();

    eprintln!("serializing run_data");
    // save config files to disk
    ops::serialize(&*plots, &root_path.join("model_plots").with_extension("config"));
    ops::serialize(&*console, &root_path.join("model_console").with_extension("config"));
}

/// Enum of all the model variants
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy)]
pub enum Models {
    BASELINE,
}

impl std::fmt::Display for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Models::BASELINE => write!(f, "baseline"),
        }
    }
}

impl Default for Models {
    fn default() -> Self {
        Models::BASELINE
    }
}

/// Send Runs to UI
#[derive(Resource, Deref, DerefMut, Clone)]
pub struct RunSend(Sender<RunId>);

/// Receive Runs from UI
#[derive(Resource, Deref, DerefMut)]
pub struct RunRecv(Receiver<RunId>);

/// A struct which fully identifies the model
pub struct RunId(pub Models, pub RunInfo, pub Entity);

/// This struct represents an individual training run, it has the information to restart itself
#[derive(Serialize, Deserialize, Default, Clone, Component)]
pub struct RunInfo {
    pub config: Config,             // TODO: convert types to this more easily,
    pub model_class: String, // name for the class of models this falls under
    pub version: usize,      // id for this run
    pub comments: String,
    pub dataset: String,
    pub err_status: Option<String>, // True is returned successfully, false if Killed mid-run
    // pub checkpoints: Vec<(f32, std::path::PathBuf)>, // (step, path)
}

impl RunInfo {
    pub fn run_name(&self) -> String {
        format!("{}-v{}", self.model_class, self.version)
    }

    // pub fn add_checkpoint(&mut self, step: f32, path: std::path::PathBuf) {
    //     self.checkpoints.push((step, path));
    // }

    // pub fn get_checkpoint(&self, i: usize) -> Option<std::path::PathBuf> {
    //     self.checkpoints.get(i).and_then(|x| Some(x.1.clone()))
    // }

    pub fn show_basic(&self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            if self.comments.len() > 0 {
                ui.collapsing("comments", |ui| {
                    ui.label(&self.comments);
                });
            }

            // ui.collapsing("checkpoints", |ui| {
            //     egui::ScrollArea::vertical().id_source("click checkpoints").show(ui, |ui| {

            //         for (j, checkpoint) in self.checkpoints.iter() {
            //             // TODO: when checkpoint is clicked, show loss as well
            //             ui.horizontal(|ui| {
            //                 ui.label(format!("step {}", j));
            //                 ui.label(checkpoint.to_str().unwrap());
            //             });
            //         }
            //     });
            // });
            if self.err_status.is_some() {
                ui.label(format!("error status: {:?}", self.err_status));
            }
            ui.label(format!("dataset: {}", self.dataset));
            ui.label(format!("model class: {}", self.model_class));
            ui.collapsing("run configs", |ui| {
                super::immutable_show(&self.config, ui);
            });
        });

    }
}

/// Tracking performance, memory usage, etc.
#[derive(Resource, Default)]
pub struct RunStats {
    runs: HashMap<Entity, models::RunStats>
}

impl RunStats {
    pub fn has_stat(&self, id: Entity) -> bool {
        self.runs.contains_key(&id)
    }

    pub fn update(&mut self, id: Entity, stats: models::RunStats) {
        self.runs.insert(id, stats);
    }

    pub fn show_basic_stats(&self, id: Entity, ui: &mut egui::Ui) {
        if let Some(stat) = self.runs.get(&id) {
            if let Some(step_time) = stat.step_time {
                ui.label(format!("step time {:.5}s", step_time));
            }
        }
    }
}

/// Since each run is identified with an Entity, sending a Kill event for a particular entity
/// should kill it. Listeners for each run type should listen for this event, and kill their
/// respective runs when this event is heard.
#[derive(Deref)]
pub struct Kill(pub Entity);

/// Once the listener kills the task, this Event is sent back to RunQueue to confirm that
/// it is alright to free its resources.
#[derive(Deref)]
pub struct Despawn(pub Entity);

pub type SpawnRun = Box<dyn FnOnce(&mut Commands) -> Result<Entity> + Send + Sync>;
/// A wrapper with all of the required information to spawn a new run
pub struct Spawn(pub RunInfo, pub SpawnRun);


/// ModelPlots contains the various plots of the model
#[derive(Resource, Default, Serialize, Deserialize)]
pub struct ModelPlots {
    graphs: HashMap<String, PlotGraph>, // each key represents the title of the plot
}

impl ModelPlots {
    /// Each plot has a title, and under each title, multiple lines are graphed, by run
    /// Inserts a new plot with title if there is none
    /// Appends onto existing run if name and title are pre-existing, else creates a new run
    pub fn add_plot(&mut self, title: &str, run_name: &str, x: f32, y: f32) {
        if let Some(graph) = self.graphs.get_mut(title) {
            if let Some(run) = graph.plots.get_mut(run_name) {
                run.push((x, y));
            } else {
                graph.plots.insert(run_name.into(), vec![(x, y)]);
            }
        } else {
            // TODO: kind of ad-hoc
            let (x_title, y_title) = if title.contains("accuracy") {
                ("steps", "accuracy")
            } else if title.contains("loss") {
                ("steps", "loss")
            } else {
                ("", "")
            };

            self.graphs.insert(
                title.into(),
                PlotGraph {
                    title: title.into(),
                    x_title: x_title.into(),
                    y_title: y_title.into(),
                    plots: HashMap::from([(run_name.to_string(), vec![(x, y)])]),
                },
            );
        }
    }
}

/// Each PlotGraph signifies a graph which contains multiple lines, from multiple runs
#[derive(Deserialize, Serialize, Default)]
pub struct PlotGraph {
    title: String,
    x_title: String,
    y_title: String,
    plots: HashMap<String, Vec<(f32, f32)>>,
}

#[derive(Resource, Serialize, Deserialize)]
pub struct Console {
    pub console_msgs: VecDeque<String>,
    pub max_console_msgs: usize,
}

impl Console {
    pub fn new(n_logs: usize) -> Self {
        Console {
            console_msgs: VecDeque::new(),
            max_console_msgs: n_logs,
        }
    }

    pub fn log(&mut self, msg: String) {
        self.console_msgs.push_front(msg);
        if self.console_msgs.len() > self.max_console_msgs {
            self.console_msgs.pop_back();
        }
    }

    pub fn console_ui(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for text in &self.console_msgs {
                ui.label(text);
            }
        });
    }
}

impl Default for Console {
    fn default() -> Self {
        Self {
            console_msgs: VecDeque::new(),
            max_console_msgs: 50,
        }
    }
}