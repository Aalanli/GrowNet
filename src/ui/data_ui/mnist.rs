use bevy_egui::egui;
use anyhow::Result;

use crate::datasets::{MnistParams, DatasetTypes};
use super::super::data_ui;
use super::{Param, DatasetSetup, DatasetBuilder, viewers::ClassificationViewer};


pub struct MNIST;
impl DatasetSetup for MNIST {
    fn parameters() -> Box<dyn DatasetBuilder> {
        Box::new(MnistParams::default())
    }

    fn viewer() -> Box<dyn data_ui::ViewerUI> {
        Box::new(ClassificationViewer::default())
    }

    fn transforms() -> Vec<super::TransformTypes> {
        vec![]
    }

    fn name() -> &'static str {
        "mnist"
    }
}

impl Param for MnistParams {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.path).hint_text("dataset path"));
                ui.label("batch size");
                ui.add(egui::DragValue::new(&mut self.batch_size));
            });
        });
    }
    fn config(&self) -> String {
        ron::to_string(&self).unwrap()
    }
    fn load_config(&mut self, config: &str) {
        let new_self: Self = ron::from_str(config).unwrap();
        *self = new_self;
    }
}

/// Each dataset supplies its own setup method through its parameters, through egui.
impl DatasetBuilder for MnistParams {
    fn build(&self) -> Result<DatasetTypes> {
        self.build()
    }
}
