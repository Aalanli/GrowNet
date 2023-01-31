use std::marker::PhantomData;

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tch::Tensor;

use super::{data, Transform};
use crate::ops;
use crate::Config;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Normalize {
    pub mu: f32,
    pub range: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Normalize {
            mu: 0.0,
            range: 2.0,
        }
    }
}

impl Transform<data::Image, data::Image> for Normalize {
    fn transform(&mut self, mut data: data::Image) -> data::Image {
        let mut min = data.image[[0, 0, 0, 0]];
        let mut max = data.image[[0, 0, 0, 0]];
        data.image.for_each(|x| {
            min = min.min(*x);
            max = max.max(*x);
        });
        let width = self.range / (max - min);
        let center = (max + min) / 2.0;
        data.image.mapv_inplace(|x| (x - center + self.mu) / width);
        data
    }
}

impl Transform<Tensor, Tensor> for Normalize {
    fn transform(&mut self, x: Tensor) -> Tensor {
        let min = x.min();
        let max = x.max();
        let width = self.range / (&max - &min);
        let center = (max + min) / 2.0;
        (x - center + Tensor::from(self.mu)) / width
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BasicImAugumentation {
    pub flip: bool,
    pub crop: i64,
    pub cutout: i64,
}

impl BasicImAugumentation {
    pub fn transform(&self, x: &Tensor) -> Tensor {
        tch::vision::dataset::augmentation(x, self.flip, self.crop, self.cutout)
    }
}

impl Default for BasicImAugumentation {
    fn default() -> Self {
        Self {
            flip: true,
            crop: 4,
            cutout: 8,
        }
    }
}

impl Transform<data::Image, data::Image> for BasicImAugumentation {
    fn transform(&mut self, data: data::Image) -> data::Image {
        let ts = ops::convert_image_array(&data.image.view()).unwrap();
        let ts = Self::transform(self, &ts);
        let im = ops::convert_image_tensor(&ts).unwrap();
        data::Image { image: im }
    }
}

impl Transform<Tensor, Tensor> for BasicImAugumentation {
    fn transform(&mut self, x: Tensor) -> Tensor {
        Self::transform(self, &x)
    }
}
