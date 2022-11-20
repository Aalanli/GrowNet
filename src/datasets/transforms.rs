
use std::marker::PhantomData;

use serde::{Serialize, de::DeserializeOwned, Deserialize};
use bevy_egui::egui;

use crate::ui::Param;
use super::ImageDataPoint;


pub trait Transform: Param {
    type DataPoint;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint;
}

/// Simple generic transform implementation, to avoid implementing
/// the necessary traits for each possible state
#[derive(Clone)]
pub struct SimpleTransform<DataPoint: Send + Sync, T: Serialize + DeserializeOwned + Send + Sync + Clone> {
    pub state: T,
    pub transform_fn: fn(&T, DataPoint) -> DataPoint,
    pub ui_fn: fn(&mut T, &mut egui::Ui),
}

impl<D: Send + Sync, T: Serialize + DeserializeOwned + Send + Sync + Clone> Param for SimpleTransform<D, T> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        let ui_fn = self.ui_fn;
        ui_fn(&mut self.state, ui);
    }
    fn config(&self) -> String {
        ron::to_string(&self.state).unwrap()
    }
    fn load_config(&mut self, config: &str) {
        self.state = ron::from_str(config).unwrap();
    }
}

impl<D: Send + Sync, T: Serialize + DeserializeOwned + Send + Sync + Clone> Transform for SimpleTransform<D, T> {
    type DataPoint = D;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint {
        let t_fn = self.transform_fn;
        t_fn(&self.state, data)
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Identity<D: Sync + Send>(PhantomData<D>);

impl<D: Sync + Send> Param for Identity<D> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Identity");
    }

    fn config(&self) -> String {
        "".to_string()
    }

    fn load_config(&mut self, _config: &str) {}
}

impl<D: Sync + Send> Transform for Identity<D> {
    type DataPoint = D;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint {
        data
    }
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

impl Param for Normalize {
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
    fn config(&self) -> String {
        ron::to_string(self).unwrap()
    }
    fn load_config(&mut self, config: &str) {
        *self = ron::from_str(config).unwrap();
    }
}

impl Transform for Normalize {
    type DataPoint = ImageDataPoint;
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