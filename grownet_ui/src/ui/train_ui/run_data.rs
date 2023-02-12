use std::collections::{HashMap, VecDeque};

use anyhow::{Error, Result};
use bevy::prelude::Resource;
use bevy_egui::egui;
use serde::{Deserialize, Serialize};

pub use model_lib::{models, Config};
pub use models::TrainProcess;

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
