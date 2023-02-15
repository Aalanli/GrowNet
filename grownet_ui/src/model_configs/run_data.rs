use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::*;
use bevy_egui::egui;
use serde::{Deserialize, Serialize};

pub use model_lib::{models, Config};
pub use models::{TrainProcess, TrainRecv, TrainSend};

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


/// The (unified?) Log which visualization utilities consume
#[derive(Clone)]
pub enum Log {
    PLOT(String, String, f32, f32),
    CONSOLE(String),
    ERROR(String),
}

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
        self.console_msgs.push_back(msg);
        if self.console_msgs.len() > self.max_console_msgs {
            self.console_msgs.pop_front();
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

pub fn handle_baseline_logs(
    plots: &mut ModelPlots,
    console: &mut Console,
    logs: &[Log],
) -> Result<()> {
    let mut some_err = Ok(());
    for log in logs {
        match log.clone() {
            Log::PLOT(name, run_name, x, y) => {
                plots.add_plot(&name, &run_name, x, y);
            }
            Log::CONSOLE(msg) => {
                console.log(msg);
            }
            Log::ERROR(err_msg) => {
                some_err = Err(Error::msg(err_msg));
            }
        }
    }
    some_err
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

    pub fn add_checkpoint(&mut self, step: f32, path: std::path::PathBuf) {
        self.checkpoints.push((step, path));
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
                        super::immutable_show(&self.config, ui);
                    });
                });

            });
        });
    }
}


#[derive(Deref, DerefMut, Serialize, Deserialize, Clone, Resource)]
pub struct ModelRunInfo(HashMap<Models, HashMap<String, RunInfo>>);

impl ModelRunInfo {
    pub fn add_info(&mut self, model: Models, name: String, info: RunInfo) -> Result<()> {
        if !self.contains_key(&model) {
            self.insert(model, HashMap::new());
        }

        let class = self.get_mut(&model).unwrap();
        if class.contains_key(&name) {
            return Err(Error::msg(format!("there is already a run of name {name}")));
        } else {
            class.insert(name, info);
        }
        Ok(())
    }

    pub fn show_past_runs(&self, model: Models, ui: &mut egui::Ui) {
        /*
        
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
         */
    }
}
