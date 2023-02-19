use bevy_egui::egui;
use bevy::prelude::Component;

use crate::UI;
use model_lib::{Config, Options};

mod run_data;
mod plots;
pub mod baseline;

pub use run_data::{
    RunDataPlugin,     // registers various caches into bevy
    Models,            // all possible model variants
    Console,           // a console showing raw info
    RunInfo,           // the details of a run
    RunId,             // The model, the runinfo, the Entity, used to send finished runs to train_ui
    RunSend,           // A channel to send RunIds
    RunRecv,           // A channel to receive RunIds
    Kill,              // A bevy event sent from the Ui, to kill a particular run, associated with an Entity
    Despawn,           // A confirmation from the system that the run has been killed
    Spawn,             // A pair containing the runinfo and a function to spawn the necessary elements to initiate a training run
    SpawnRun,          // A type alias for Box<dyn FnOnce(&mut Commands) -> Result<Entity> + Send + Sync>, the spawning function
    RunStats,          // A struct containing runtime info, such as step time and memory usage
    DefaultPlotViewer, // Main PlotViewer, there could be multiplex
};

pub use plots::{
    ModelPlots,    // The primary cache from all model runs
    PlotLine,      // A Vec<(f64, f64)> representing (x, y) coordinates, where x is monotonically increasing
    PlotId,        // A unique identifier for each line
    PlotViewer,    // The Ui to show the plots
};

impl UI for Config {
    fn ui(&mut self, ui: &mut egui::Ui) {
        for (k, v) in self.iter_mut() {
            match v {
                Options::BOOL(i) => {
                    ui.checkbox(i, k);
                }
                Options::INT(i) => {
                    ui.horizontal(|ui| {
                        ui.label(k);
                        ui.add(egui::DragValue::new(i).speed(0.1));
                    });
                }
                Options::FLOAT(i) => {
                    ui.horizontal(|ui| {
                        ui.label(k);
                        ui.add(egui::DragValue::new(i).speed(0.1));
                    });
                }
                Options::STR(i) => {
                    ui.add(egui::TextEdit::singleline(i).hint_text(k));
                }
                Options::PATH(i) => {
                    let mut str = i.to_str().unwrap().to_string();
                    ui.add(egui::TextEdit::singleline(&mut str).hint_text(k));
                    *i = str.into();
                }
                Options::CONFIG(c) => {
                    ui.horizontal(|ui| {
                        // indent
                        ui.label("  ");
                        ui.vertical(|ui| {
                            egui::CollapsingHeader::new(k)
                                .default_open(true)
                                .show(ui, |ui| {
                                    c.ui(ui);
                                });
                        });
                    });
                }
            }
        }
    }
}

/// Only show through the ui, don't change anything
pub fn immutable_show(config: &Config, ui: &mut egui::Ui) {
    for (k, v) in config.iter() {
        match v {
            Options::BOOL(i) => {
                ui.label(format!("{k}: {i}"));
            }
            Options::INT(i) => {
                ui.label(format!("{k}: {i}"));
            }
            Options::FLOAT(i) => {
                ui.label(format!("{k}: {i}"));
            }
            Options::STR(i) => {
                ui.label(format!("{k}: {i}"));
            }
            Options::PATH(i) => {
                ui.label(format!("{k}: {}", i.to_str().unwrap()));
            }
            Options::CONFIG(c) => {
                ui.horizontal(|ui| {
                    // indent
                    ui.label("  ");
                    ui.vertical(|ui| {
                        egui::CollapsingHeader::new(k)
                            .default_open(true)
                            .show(ui, |ui| {
                                immutable_show(c, ui);
                            });
                    });
                });
            }
        }
    }
}

pub struct ConfigUiWrapper {
    config: Config,

}