use serde::{Serialize, Deserialize};
use strum::{IntoEnumIterator, EnumIter};
use bevy_egui::egui;

use super::{ImClassifyDataPoint, ImageDataPoint};


#[derive(Debug, Serialize, Deserialize, EnumIter)]
pub enum Transforms {
    Normalize
}

pub enum TransformType {
    Image(Box<dyn ImageTransform>)
}

pub trait ImageTransform: Send + Sync {
    fn transform(&mut self, data: ImageDataPoint) -> ImageDataPoint;
}

pub trait TransformUI {
    fn ui(&mut self, ui: &egui::Ui);
    fn build(&self) -> TransformType;
}

pub struct Normalize {
    mu: f32,
    range: f32,
}
