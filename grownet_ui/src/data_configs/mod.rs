/// Defining the config adjustment through the ui
use ndarray::prelude::*;
use anyhow::Result;
use bevy_egui::egui;

use crate::UI;
use model_lib::{datasets, Config};
use datasets::transforms::Transform;

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
                ui.label("transform params");
                //self.transform.ui(ui);
                self.path = path_str.into();
            });
        });
    }
}

impl UI for datasets::transforms::Normalize {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("Normalize transform params");
            ui.vertical(|ui| {
                ui.label("mu");
                ui.add(egui::DragValue::new(&mut self.mu).speed(0.01));
                ui.label("range");
                ui.add(egui::DragValue::new(&mut self.range).speed(0.01));
            });
        });
    }
}

impl UI for datasets::transforms::BasicImAugumentation {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.flip, "flip");
                ui.label("crop");
                ui.add(egui::DragValue::new(&mut self.crop));
                ui.label("cutout");
                ui.add(egui::DragValue::new(&mut self.cutout));                
            });
        });
    }
}

impl<F, In, Out> UI for datasets::transforms::FnTransform<F, In, Out> 
where F: Send + Sync, In: Send + Sync, Out: Send + Sync 
{
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("stateless function");
    }
}

impl<T1, T2> UI for datasets::transforms::Compose<T1, T2>
where T1: Transform + UI, T2: Transform + UI
{
    fn ui(&mut self, ui: &mut egui::Ui) {
        self.t1.ui(ui);
        self.t2.ui(ui);
    }
}

impl<T1, T2, F, O> UI for datasets::transforms::Concat<T1, T2, F, O>
where T1: Transform + UI, T2: Transform + UI, F: Fn(&T1, &T2, T1::In) -> O + Send + Sync + Clone,
    O: Send + Sync + Clone
{
    fn ui(&mut self, ui: &mut egui::Ui) {
        self.t1.ui(ui);
        self.t2.ui(ui);
    }
}