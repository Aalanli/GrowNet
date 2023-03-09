use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, Range};

use itertools::Itertools;
use crossbeam::channel::{Sender, Receiver};
use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
use plotters::coord::types::RangedCoordf64;
use plotters::style::Color;
use serde::{Deserialize, Serialize};

pub use model_lib::{models, Config};
pub use models::{TrainProcess, TrainRecv, TrainSend, PlotPoint};
pub use crate::ui::OperatingState;
pub use super::{ModelPlots, PlotId, PlotViewerV1, PlotViewerV2};

use crate::{ops, Serializer};

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
            // .insert_resource(PlotViewerV1::default())
            .insert_resource(PlotViewerV2::default())
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
    // mut plot_viewer: ResMut<PlotViewerV1>,
    mut plot_viewer2: ResMut<PlotViewerV2>,
    mut console: ResMut<Console>,
    serializer: Res<Serializer>
) {
    serializer.deserialize("model_plots", &mut *plots);
    serializer.deserialize("model_console", &mut *console);
    // serializer.deserialize("plot_viewer", &mut *plot_viewer);
    serializer.deserialize("plot_viewer2", &mut *plot_viewer2);
}

/// write run data to disk
fn save_run_data(
    plots: Res<ModelPlots>,
    // plot_viewer: Res<PlotViewerV1>,
    plot_viewer2: Res<PlotViewerV2>,
    console: Res<Console>,
    mut serializer: ResMut<Serializer>
) {
    serializer.serialize("model_plots", &*plots);
    serializer.serialize("model_console", &*console);
    // serializer.serialize("plot_viewer", &*plot_viewer);
    serializer.serialize("plot_viewer2", &*plot_viewer2);
}

/// Enum of all the model variants
#[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Debug)]
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
                super::config_ui_show(&self.config, ui);
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
