
use serde::{Serialize, Deserialize};
use strum::{IntoEnumIterator, EnumIter};
use bevy_egui::egui;

use super::{ImClassifyDataPoint, ImageDataPoint};


pub trait Transform: Send + Sync {
    type DataPoint;
    fn ui_setup(&mut self, ui: &mut egui::Ui);
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint;
}


#[derive(Debug, Serialize, Deserialize)]
pub struct Normalize {
    mu: f32,
    range: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Normalize { mu: 0.0, range: 1.0 }
    }
}

impl Transform for Normalize {
    type DataPoint = ImageDataPoint;
    fn ui_setup(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label("mu");
                ui.add(egui::DragValue::new(&mut self.mu));
                ui.label("range");
                ui.add(egui::DragValue::new(&mut self.range));
            });
        });
    }

    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint {
        data
    }
}