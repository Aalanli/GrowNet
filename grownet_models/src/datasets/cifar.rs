use std::fs;
use std::path;
use rand::seq::SliceRandom;

use ndarray::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Error, Context};

use super::{Dataset, DatasetBuilder};
use super::data::{ImClassify, Image};

/// A struct containing image data, where all images are are the same size
/// and stored contiguously next to each other
struct ContiguousStaticImage {
    images: Vec<f32>,
    width: usize,
    height: usize,
    offset: usize, // width * height * 3 
    num_images: usize
}

impl ContiguousStaticImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { images: Vec::new(), width, height, offset: width * height * 3, num_images: 0 }
    }
    pub fn push_images(&mut self, buf: &[f32]) {
        assert!(buf.len() % self.offset == 0);
        self.images.extend_from_slice(buf);
        self.num_images += buf.len() / self.offset;
    }
    pub fn extend_images<T: Iterator<Item = f32>>(&mut self, iter: T) {
        let mut count = 0;
        for h in iter {
            self.images.push(h);
            count += 1;
        }
        self.num_images += count / self.offset;
    }
    fn reserve(&mut self, images: usize) {
        self.images.reserve_exact(images * self.offset);
    }

    pub fn index(&self, indices: &[usize]) -> Image {
        let buf_sz = indices.len() * self.offset;
        let mut buf = Vec::<f32>::new();
        buf.reserve_exact(buf_sz);
        unsafe { buf.set_len(buf_sz); }

        for (i, j) in (0..indices.len()).zip(indices.iter()) {
            let j = *j;
            buf[i*self.offset..(i+1)*self.offset].copy_from_slice(&self.images[j*self.offset..(j+1)*self.offset]);
        }

        let arr = Array::from_shape_vec((indices.len(), self.width, self.height, 3), buf).unwrap();
        Image { image: arr }
    }
}

// main parameters driving the cifar10 dataset
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Cifar10Params {
    path: path::PathBuf,
    train_batch_size: usize,
    test_batch_size: usize,
}

impl DatasetBuilder for Cifar10Params {
    type Dataset = Cifar10;

    fn build_train(&self) -> anyhow::Result<Self::Dataset> {
        if self.train_batch_size == 0 {
            return Err(Error::msg("batch size cannot be zero"));
        }
        Cifar10::build_train(&self.path, self.train_batch_size)
    }

    fn build_test(&self) -> Option<anyhow::Result<Self::Dataset>> {
        if self.test_batch_size == 0 {
            return Some(Err(Error::msg("batch size cannot be zero")));
        }
        Some(Cifar10::build_test(&self.path, self.test_batch_size))
    }
}

pub struct Cifar10 {
    images: ContiguousStaticImage,
    labels: Vec<u32>,
    label_names: Vec<String>,
    shuffle: Vec<usize>,
    idx: usize,
    batch_size: usize,
}

impl Cifar10 {
    fn insert_images(raw_buf: Vec<u8>, labels: &mut Vec<u32>, images: &mut ContiguousStaticImage) {
        for i in 0..10000 {
            let offset = i * 3073;
            labels.push(raw_buf[offset] as u32);
            let iter = (0..1024).map(|j| {
                let buf_ref = &raw_buf;
                let temp = (0..3).map(move |k| {
                    buf_ref[offset + 1 + j + k * 1024] as f32 / 255.0 // normalize between [0, 1]
                });
                temp  
            }).flatten();
            images.extend_images(iter);
        }
    }

    pub fn build_test(data_folder: &path::Path, batch_size: usize) -> Result<Self> {
        let test_file = "test_batch.bin";
        let raw_buf: Vec<u8> = fs::read(data_folder.join(test_file)).context("test batch file")?;

        let mut test_images = ContiguousStaticImage::new(32, 32);
        let mut test_labels = Vec::<u32>::new();

        test_labels.reserve_exact(10000);
        test_images.reserve(10000);
        Self::insert_images(raw_buf, &mut test_labels, &mut test_images);

        let class_names = fs::read_to_string(data_folder.join("batches.meta.txt")).context("meta file")?;
        let label_names: Vec<String> = class_names.split("\n").map(|x| x.to_string()).collect();

        let test_shuffle = (0..10000).collect();
        let x = Self { images: test_images, labels: test_labels, label_names, shuffle: test_shuffle, idx: 0, batch_size };
        Ok(x)
    }

    pub fn build_train(data_folder: &path::Path, batch_size: usize) -> Result<Self> {
        let train_files = (1..6).map(|x| format!("data_batch_{x}.bin"))
            .map(|f| data_folder.join(f)).filter(|f| f.exists());

        let num_images = 50000;
        let mut train_images = ContiguousStaticImage::new(32, 32);
        let mut train_labels = Vec::<u32>::new();
        train_labels.reserve_exact(num_images);
        train_images.reserve(num_images);

        for file in train_files {
            let raw_buf: Vec<u8> = fs::read(&file)?;
            Self::insert_images(raw_buf, &mut train_labels, &mut train_images);
        }

        let class_names = fs::read_to_string(data_folder.join("batches.meta.txt")).context("meta file")?;
        let label_names: Vec<String> = class_names.split("\n").map(|x| x.to_string()).collect();

        let train_shuffle = (0..num_images).collect();

        let x = Self { images: train_images, labels: train_labels, label_names, shuffle: train_shuffle, idx: 0, batch_size };
        Ok(x)
    }
}


impl Dataset for Cifar10 {
    type DataPoint = ImClassify;

    fn next(&mut self) -> Option<Self::DataPoint> {
        // batch size stride greater than the number of elements
        if self.idx + self.batch_size >= self.shuffle.len() {
            return None;
        }
        let indices: Vec<_> = (self.idx..self.batch_size + self.idx).map(|i| self.shuffle[i]).collect();
        let img_datapoint = self.images.index(&indices);
        let labels = (self.idx..self.batch_size + self.idx).map(|x| self.labels[self.shuffle[x]]).collect();
        self.idx += self.batch_size;

        let img = ImClassify { image: img_datapoint, label: labels };
        Some(img)
    }

    fn reset(&mut self) {
        self.idx = 0;
    }

    fn shuffle(&mut self) {
        self.shuffle.shuffle(&mut rand::thread_rng());
    }
}
