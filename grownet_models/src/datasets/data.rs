/////////////////////////////////////////////////////////////////////////////////////////
/// The Data types that the datasets output and transforms input.
use crate::ops;
use anyhow::Result;
use ndarray::prelude::*;

/// Expect images to be normalized between [0, 1] and has a shape of NWHC
#[derive(Clone)]
pub struct Image {
    pub image: Array4<f32>,
}


/// The data point associated with the image detection task, this is the type which
/// gets fed into the model
#[derive(Clone)]
pub struct ImClassify {
    pub image: Image,
    pub label: Vec<u32>,
}

// pub struct ObjDetectionDataPoint;
// pub struct TextGenerationDataPoint;
// pub struct ImageModelingDataPoint;

use image::DynamicImage;
impl Image {
    pub fn size(&self) -> [usize; 2] {
        let shape = self.image.dim();
        [shape.1, shape.2]
    }

    pub fn from_image(im: &DynamicImage) -> Image {
        let im = im.to_rgb32f();
        let w = im.width();
        let h = im.height();
        let buf = im.into_raw();
        let array = Array::from_shape_vec((1, w as usize, h as usize, 3), buf).unwrap();
        Self { image: array }
    }
}
