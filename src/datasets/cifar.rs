use super::{DatasetUI, Dataset, ImClassifyDataPoint};
use crate::ui::Config;

pub struct Cifar10Params {}

impl Default for Cifar10Params {
    fn default() -> Self {
        Self {  }
    }
}

impl Config for Cifar10Params {
    fn config(&self) -> String {
        todo!()
    }

    fn load_config(&mut self, config: &str) {
        todo!()
    }
}

impl DatasetUI for Cifar10Params {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        todo!()
    }

    fn build(&self) -> anyhow::Result<super::DatasetTypes> {
        todo!()
    }
}

pub struct Cifar10 {}

impl Dataset for Cifar10 {
    type DataPoint = ImClassifyDataPoint;

    fn next(&mut self) -> Option<Self::DataPoint> {
        todo!()
    }

    fn reset(&mut self) {
        todo!()
    }

    fn shuffle(&mut self) {
        todo!()
    }
}