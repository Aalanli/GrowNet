use bevy_egui::egui;

use model_lib::{Config, Options};
use crate::UI;

pub mod baseline;

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
                },
                Options::FLOAT(i) => {
                    ui.horizontal(|ui| {
                        ui.label(k);
                        ui.add(egui::DragValue::new(i).speed(0.1));
                    });
                },
                Options::STR(i) => {
                    ui.add(egui::TextEdit::singleline(i).hint_text(k));
                },
                Options::PATH(i) => {
                    let mut str = i.to_str().unwrap().to_string();
                    ui.add(egui::TextEdit::singleline(&mut str).hint_text(k));
                    *i = str.into();
                },
                Options::CONFIG(c) => {
                    ui.horizontal(|ui| {
                        // indent
                        ui.label("  ");
                        ui.vertical(|ui| {
                            egui::CollapsingHeader::new(k).default_open(true).show(ui, |ui| {
                                c.ui(ui);
                            });
                        });
                    });
                },
                
            }
        }
    }
}
