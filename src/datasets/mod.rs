/// This module only defines the dataset logic for loading and processing datasets
/// This is separate from the data_ui module, which deals with integrating with the ui
/// for visualizations, etc.
/// 
/// Separating the logic can enable headlessmode which will be for future work

use ndarray::prelude::*;
use anyhow::Result;
use crate::Param;

pub mod transforms;

mod mnist;
pub use mnist::MnistParams;
pub mod cifar;


use self::transforms::Transform;

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Option<Self::DataPoint>;
    fn reset(&mut self);
    fn shuffle(&mut self);
}

pub trait DatasetBuilder: Param {
    type Dataset: Dataset;
    fn build_train(&self) -> Result<Self::Dataset>;
    fn build_test(&self) -> Option<Result<Self::Dataset>>;
}

pub trait DataTransforms: Param {
    type DataPoint;
    fn transform(&self, data: Self::DataPoint) -> Self::DataPoint;
}

/// This is the unification of possible Datasets behaviors, constructed from DatasetUI, or
/// some other parameter-adjusting setup trait.
pub type ClassificationType = Box<dyn Dataset<DataPoint = ImClassifyDataPoint>>;
pub enum DatasetTypes {
    Classification(ClassificationType),
    // Detection(Box<dyn ImDetection>),
    // Generation(Box<dyn ImGeneration>),
}

/// Transform type enum, reflective of the DatasetTypes enum, which only depends
/// on the type of the output data point
pub type ClassificationTransform = Box<dyn transforms::Transform<DataPoint = ImClassifyDataPoint>>;

pub enum TransformTypes {
    Identity,
    Classification(ClassificationTransform)
}

impl Clone for TransformTypes {
    fn clone(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Classification(arg0) => Self::Classification(dyn_clone::clone_box(&**arg0)),
        }
    }
}

pub struct DataWrapper<D> {
    dataset: Box<dyn Dataset<DataPoint = D>>,
    transform: Box<dyn Transform<DataPoint= D>>
}

impl<D> DataWrapper<D> {
    pub fn new(dataset: Box<dyn Dataset<DataPoint = D>>, transform: Box<dyn Transform<DataPoint= D>>) -> Self {
        DataWrapper { dataset, transform }
    }
}

impl<D> Iterator for DataWrapper<D> {
    type Item = D;
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.dataset.next()?;
        Some(self.transform.transform(x))
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
/// The Data types that the datasets output and transforms input.
#[derive(Clone)]
pub struct ImageDataPoint {
    pub image: Array4<f32>
}

/// The data point associated with the image detection task, this is the type which
/// gets fed into the model
#[derive(Clone)]
pub struct ImClassifyDataPoint {
    pub image: ImageDataPoint,
    pub label: Vec<u32>
}

// pub struct ObjDetectionDataPoint;
// pub struct TextGenerationDataPoint;
// pub struct ImageModelingDataPoint;

use image::DynamicImage;
impl ImageDataPoint {
    pub fn size(&self) -> [usize; 2] {
        let shape = self.image.dim();
        [shape.1, shape.2]
    }

    pub fn from_image(im: &DynamicImage) -> ImageDataPoint {
        let im = im.to_rgb32f();
        let w = im.width();
        let h = im.height();
        let buf = im.into_raw();
        let array = Array::from_shape_vec((1, w as usize, h as usize, 3), buf).unwrap();
        Self { image: array }
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
