use std::{path::PathBuf, str::FromStr};
use rand::seq::SliceRandom;

use ndarray::prelude::*;
use bevy_egui::egui;
use itertools::Itertools;
use serde::{Serialize, Deserialize};
use image::io::Reader as ImageReader;
use anyhow::{Context, Result, Error};


use crate::UI;
use super::{Dataset, DatasetBuilder};
use super::{ImageDataPoint, ImClassifyDataPoint};


/// Main configuration parameters for Mnist
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MnistParams {
    path: PathBuf,
    train_batch_size: usize,
    test_batch_size: usize,
}

impl UI for MnistParams {
    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                let mut path_str = self.path.to_str().unwrap().to_owned();
                ui.label("Cifar10 Parameters");
                ui.text_edit_singleline(&mut path_str);
                ui.label("train batch size");
                ui.add(egui::DragValue::new(&mut self.train_batch_size));
                ui.label("test batch size");
                ui.add(egui::DragValue::new(&mut self.test_batch_size));
                self.path = path_str.into();
            });
        });
    }
}

impl DatasetBuilder for MnistParams {
    type Dataset = Mnist;

    fn build_train(&self) -> anyhow::Result<Self::Dataset> {
        if self.train_batch_size == 0 {
            return Err(Error::msg("batch size cannot be zero"));
        }
        MnistParams::build_train(self)
    }

    fn build_test(&self) -> Option<anyhow::Result<Self::Dataset>> {
        if self.test_batch_size == 0 {
            return Some(Err(Error::msg("batch size cannot be zero")));
        }
        Some(MnistParams::build_test(self))
    }
}


impl MnistParams {
    fn read_subdirs(dir: &PathBuf) -> Result<Vec<(u32, PathBuf)>, Error> {
        let mut correct_paths = Vec::<(u32, PathBuf)>::new();

        for path in dir.read_dir()? {
            let a = path?;
            let label = a.file_name().into_string().unwrap().parse::<u32>()
                .with_context(|| format!("Dir {} is not a numeral", a.path().to_str().unwrap()))?;
            let subpaths = a.path().read_dir().with_context(|| format!("No dir {} exists mnist", dir.to_str().unwrap()))?;
            for im_file in subpaths {
                if let Ok(file) = im_file {
                    correct_paths.push((label, file.path()));
                }
            }
        }
        Ok(correct_paths)
    }
    pub fn build_test(&self) -> Result<Mnist> {
        let trainset: PathBuf = self.path.join("testing");
        let test_paths = Self::read_subdirs(&trainset)?;
        
        let mut test = ImClassifyData::new();
        test.push_raw(test_paths.iter().cloned());

        let order = 0..test.labels.len();
        
        let x = Mnist { data: test, shuffle: order.collect_vec(), idx: 0, batch_size: self.test_batch_size };        
        Ok(x)
    }

    pub fn build_train(&self) -> Result<Mnist> {
        let mut trainset: PathBuf = self.path.clone().into();
        trainset.push("training");

        let train_paths = Self::read_subdirs(&trainset)?;

        let mut train = ImClassifyData::new();
        train.push_raw(train_paths.iter().cloned());
        
        let order = 0..train.labels.len();

        let x = Mnist { data: train, shuffle: order.collect_vec(), idx: 0, batch_size: self.train_batch_size };
        Ok(x)
    }
}

pub struct Mnist {
    data: ImClassifyData,
    shuffle: Vec<usize>,
    idx: usize,
    batch_size: usize
}

impl Dataset for Mnist {
    type DataPoint = ImClassifyDataPoint;
    fn next(&mut self) -> Option<Self::DataPoint> {
        // batch size stride greater than the number of elements
        if self.idx + self.batch_size >= self.shuffle.len() {
            return None;
        }
        let img_slice: Vec<_> = (self.idx..self.batch_size + self.idx).map(|x| {
            &self.data.data[self.shuffle[x]]
        }).collect();
        let img = super::concat_im_size_eq(&img_slice);
        let labels = (self.idx..self.batch_size + self.idx).map(|x| self.data.labels[self.shuffle[x]]).collect();
        self.idx += self.batch_size;

        let img = ImClassifyDataPoint { image: img, label: labels };
        Some(img)
    }
    fn shuffle(&mut self) {
        self.shuffle.shuffle(&mut rand::thread_rng());
    }
    fn reset(&mut self) {
        self.idx = 0;
    }
}

/// The raw image data, this struct loads and stores all images in memory from the
/// paths supplied by the iterator, only suitable for small datasets
struct ImClassifyData {
    pub data: Vec<Array3<f32>>,
    pub labels: Vec<u32>,
}

impl ImClassifyData {
    fn new() -> Self {
        ImClassifyData { data: Vec::new(), labels: Vec::new() }
    }

    fn push_raw<It>(&mut self, it: It)
    where It: Iterator<Item = (u32, std::path::PathBuf)> 
    {
        for (l, path) in it {
            let img = ImageReader::open(path).unwrap().decode().unwrap();
            let img = img.into_rgb8();
            let arr = Array3::from_shape_vec((img.width() as usize, img.height() as usize, 3), img.to_vec()).unwrap();
            let arr = arr.mapv(|x| f32::from(x) / 255.0);
            self.data.push(arr);
            self.labels.push(l);
        }
    }
}


#[cfg(test)]
mod test {
    use std::fs;
    use super::*;
    #[test]
    fn read_files() {
        let paths = fs::read_dir("assets/ml_datasets/mnist_png/testing").unwrap();
        let im_paths = paths.map(|path| {
            let a = path.unwrap();
            let label = a.file_name().into_string().unwrap().parse::<u32>().unwrap();
            let subiter = a.path().read_dir().unwrap().map(move |im_file| (label, im_file.unwrap().path()));
            subiter
        }).flatten();
    
        let mut rawdata = ImClassifyData::new();
        rawdata.push_raw(im_paths);
    
        println!("{}", rawdata.data.len());
    
    }
    
    #[test]
    fn permute_axes() {
        use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};
    
        use rand::{thread_rng};
        use rand_distr::Distribution;
    
        let h: Array<f32, _> = Array::random((3, 4, 4), Normal::new(0.0, 1.0).unwrap());
        let p = h.clone().permuted_axes((1, 2, 0));
        let p = p.as_standard_layout();
    
        let k = h.as_slice().unwrap().iter().zip(p.as_slice().unwrap().iter())
            .all(|(a, b)| a == b);
        println!("{}", k);
        let hp = h.permuted_axes((1, 2, 0));
        let r = hp.iter().zip(p.as_slice().unwrap().iter())
        .all(|(a, b)| a == b);
        println!("{}", r);
    }

    #[test]
    fn slice() {
        let h = Array4::<f32>::ones((4, 3, 512, 512));
        let _p = h.slice(s![0, .., .., ..]);
    }
}