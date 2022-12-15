
use std::marker::PhantomData;

use bevy_egui::egui;

use serde::{Serialize, de::DeserializeOwned, Deserialize};
use anyhow::{Result, Context};

use crate::{Param, Config, UI};
use super::ImageDataPoint;


pub trait Transform: Param {
    type DataPoint;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint;
}

/// Simple generic transform implementation, to avoid implementing
/// the necessary traits for each possible state
#[derive(Clone)]
pub struct SimpleTransform<DataPoint: Send + Sync + Clone, T: Serialize + DeserializeOwned + Send + Sync + Clone> {
    pub state: T,
    pub transform_fn: fn(&T, DataPoint) -> DataPoint,
    pub ui_fn: fn(&mut T, &mut egui::Ui),
}

impl<D: Send + Sync + Clone, T: Serialize + DeserializeOwned + Send + Sync + Clone> UI for SimpleTransform<D, T> {
    fn ui(&mut self, ui: &mut egui::Ui) {
        let ui_fn = self.ui_fn;
        ui_fn(&mut self.state, ui);
    }
}

impl<D: Send + Sync + Clone, T: Serialize + DeserializeOwned + Send + Sync + Clone> Config for SimpleTransform<D, T> {
    fn config(&self) -> String {
        ron::to_string(&self.state).unwrap()
    }
    fn load_config(&mut self, config: &str) -> Result<()> {
        self.state = ron::from_str(config).context("Simple Transform")?;
        Ok(())
    }
}

impl<D: Send + Sync + Clone, T: Serialize + DeserializeOwned + Send + Sync + Clone> Transform for SimpleTransform<D, T> {
    type DataPoint = D;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint {
        let t_fn = self.transform_fn;
        t_fn(&self.state, data)
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

impl UI for Normalize {
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