use bevy_egui::egui;
use anyhow::{Result, Context};

use crate::{Param, Config, UI};
use crate::datasets::transforms::{Transform, SimpleTransform};
use crate::datasets::{MnistParams, DatasetTypes, TransformTypes, transforms, ImClassifyDataPoint};
use super::super::data_ui;
use super::{DatasetSetup, DatasetBuilder, viewers::ClassificationViewer};


pub struct MNIST;
impl DatasetSetup for MNIST {
    fn parameters() -> Box<dyn DatasetBuilder> {
        Box::new(MnistParams::default())
    }

    fn viewer() -> Box<dyn data_ui::ViewerUI> {
        Box::new(ClassificationViewer::default())
    }

    fn transforms() -> Vec<super::TransformTypes> {
        let ts: SimpleTransform<ImClassifyDataPoint, _> = transforms::SimpleTransform {
            state: transforms::Normalize::default(),
            transform_fn: |t, mut x| {
                x.image = t.transform(x.image);
                x
            },
            ui_fn: |t, ui| {
                t.ui(ui);
            }
        };

        vec![
            TransformTypes::Classification(Box::new(ts))
        ]
    }

    fn name() -> &'static str {
        "mnist"
    }
}

impl UI for MnistParams {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.path).hint_text("dataset path"));
                ui.label("batch size");
                ui.add(egui::DragValue::new(&mut self.batch_size));
            });
        });
    }
}

/// Each dataset supplies its own setup method through its parameters, through egui.
impl DatasetBuilder for MnistParams {
    fn build(&self) -> Result<DatasetTypes> {
        self.build()
    }
}
