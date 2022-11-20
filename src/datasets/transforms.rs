
use serde::{Serialize, Deserialize};
use strum::{IntoEnumIterator, EnumIter};
use bevy_egui::egui;

use super::{ImClassifyDataPoint, ImageDataPoint};


pub trait Transform: Send + Sync {
    type DataPoint;
    fn ui_setup(&mut self, ui: &mut egui::Ui);
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint;
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Normalize {
    mu: f32,
    range: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Normalize { mu: 0.0, range: 2.0 }
    }
}

impl Transform for Normalize {
    type DataPoint = ImageDataPoint;
    fn ui_setup(&mut self, ui: &mut egui::Ui) {
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

    fn transform(&self, mut data: Self::DataPoint) -> Self::DataPoint {
        let mut min = data.image[[0, 0, 0, 0]];
        let mut max = data.image[[0, 0, 0, 0]];
        data.image.for_each(|x| {
            min = min.min(*x);
            max = max.max(*x);
        });
        let width = self.range / (max - min);
        let center = (max + min) / 2.0;
        data.image.mapv_inplace(|x| {
            (x - center + self.mu) / width
        });
        data
    }
}