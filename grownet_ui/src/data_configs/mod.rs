/// Defining the config adjustment through the ui
use ndarray::prelude::*;
use anyhow::Result;
use bevy_egui::egui;

use crate::UI;
use model_lib::datasets;

impl UI for datasets::mnist::MnistParams {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                let mut path_str = self.path.to_str().unwrap().to_owned();
                ui.label("Cifar10 Parameters");
                ui.text_edit_singleline(&mut path_str);
                ui.label("train batch size");
                ui.add(egui::DragValue::new(&mut self.train_batch_size));
                ui.label("test batch size");
                ui.add(egui::DragValue::new(&mut self.test_batch_size));
                self.path = path_str.into();
            });
        });
    }
}

impl UI for datasets::cifar::Cifar10Params {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                let mut path_str = self.path.to_str().unwrap().to_owned();
                ui.label("Cifar10 Parameters");
                ui.text_edit_singleline(&mut path_str);
                ui.label("train batch size");
                ui.add(egui::DragValue::new(&mut self.train_batch_size));
                ui.label("test batch size");
                ui.add(egui::DragValue::new(&mut self.test_batch_size));
                self.path = path_str.into();
            });
        });
    }
}
