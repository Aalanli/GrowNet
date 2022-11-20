use std::fs;
use std::path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use anyhow::{Context, Result};

use bevy::prelude::*;
use bevy::app::AppExit;
use bevy_egui::{egui, EguiContext};
use ndarray::prelude::*;
use serde::{Serialize, Deserialize};
use strum::Display;
use strum::{IntoEnumIterator, EnumIter};

pub mod transforms;

mod mnist;
pub use mnist::MnistParams;
//pub mod cifar;

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Option<Self::DataPoint>;
    fn reset(&mut self);
    fn shuffle(&mut self);
}

/// This is the unification of possible Datasets behaviors, constructed from DatasetUI, or
/// some other parameter-adjusting setup trait.
pub type ClassificationType = Box<dyn Dataset<DataPoint = ImClassifyDataPoint>>;
pub enum DatasetTypes {
    Classification(ClassificationType),
    // Detection(Box<dyn ImDetection>),
    // Generation(Box<dyn ImGeneration>),
}

pub enum TransformTypes {
    Classification(Box<dyn transforms::Transform<DataPoint = ImClassifyDataPoint>>)
}

/// The Data types that the datasets output and transforms input.
pub struct ImageDataPoint {
    pub image: Array4<f32>
}

/// The data point associated with the image detection task, this is the type which
/// gets fed into the model
pub struct ImClassifyDataPoint {
    pub image: ImageDataPoint,
    pub label: Vec<u32>
}

// pub struct ObjDetectionDataPoint;
// pub struct TextGenerationDataPoint;
// pub struct ImageModelingDataPoint;

impl ImageDataPoint {
    pub fn size(&self) -> [usize; 2] {
        let shape = self.image.dim();
        [shape.1, shape.2]
    }
}


/// Assumes that all the 3d arrays have the same size, this function
/// stacks all the images in the first dimension. [W, H, C] -> [B, W, H, C]
fn concat_im_size_eq(imgs: &[&Array3<f32>]) -> ImageDataPoint {
    let whc = imgs[0].dim();
    let b = imgs.len();
    let mut img = Array4::<f32>::zeros((b, whc.0, whc.1, whc.2));
    for i in 0..b {
        let mut smut = img.slice_mut(s![i, .., .., ..]);
        smut.zip_mut_with(imgs[i], |a, b| {
            *a = *b;
        });
    }
    ImageDataPoint { image: img }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn serialize_test() {
        let param: mnist::MnistParams = ron::from_str(&fs::read_to_string("assets/configs/datasets/mnist.ron").unwrap()).unwrap();
        println!("{:?}", param);
        let str_repr = ron::to_string(&param).unwrap();
        println!("{}", str_repr);
    }
    
    #[test]
    fn write_test() {
        let test_file = "test_folder/test.ron";
        let msg = "hello";
        let dir = path::Path::new(test_file).parent().unwrap();
        if !dir.exists() {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(test_file, msg).unwrap();
    } 
}