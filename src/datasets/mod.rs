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
use strum::{IntoEnumIterator, EnumIter};

use crate::ui::Config;
pub mod mnist;
pub mod cifar;
pub mod transforms;


/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Option<Self::DataPoint>;
    fn reset(&mut self);
    fn shuffle(&mut self);
}

/// Each Dataset has two structs, that of the parameters it holds
/// and the data that it holds, this trait is implemented on the parameters for the dataset
/// this separation is made as the parameters are usually light and copyible, while
/// the data is not light, and require some non-negliable compute for the setup.
/// This trait adjusts the parameters, and builds the dataset on the parameters it holds.
pub trait DatasetUI: Sync + Send + Config {
    fn ui(&mut self, ui: &mut egui::Ui);
    fn build(&self) -> Result<DatasetTypes>;
}


/// This is the unification of possible Datasets behaviors, constructed from DatasetUI, or
/// some other parameter-adjusting setup trait.
pub type ClassificationType = Box<dyn Dataset<DataPoint = ImClassifyDataPoint>>;
pub enum DatasetTypes {
    Classification(ClassificationType),
    // Detection(Box<dyn ImDetection>),
    // Generation(Box<dyn ImGeneration>),
}

/// The unification of every possible Dataset supported in a single type.
#[derive(Clone, Copy, Debug, EnumIter, PartialEq, Eq, Hash)]
pub enum DatasetEnum {
    MNIST,
    CIFAR10,
}

impl DatasetEnum {
    /// the name used for the config paths
    pub fn name(&self) -> &str {
        match self {
            Self::MNIST => "mnist",
            Self::CIFAR10 => "cifar10"
        }
    }
    
    pub fn get_param(&self) -> Box<dyn DatasetUI> {
        match self {
            Self::MNIST => Box::new(mnist::MnistParams::default()),
            Self::CIFAR10 => Box::new(cifar::Cifar10Params::default())
        }
    }
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