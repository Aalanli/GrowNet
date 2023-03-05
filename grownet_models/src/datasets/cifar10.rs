use ndarray::prelude::*;
use anyhow::{Result, Error};

pub struct Cifar10 {
    train_img: Array4<u8>,
    train_label: Array1<u8>,
    test_img: Array4<u8>,
    test_label: Array1<u8>
}

impl Cifar10 {
    pub fn new(base_path: &str) -> Result<Self> {
        use cifar_ten::*;
        let CifarResult(train_data, train_labels, test_data, test_labels) = Cifar10::default()
            .download_and_extract(true)
            .base_path(base_path)
            .encode_one_hot(false)
            .build()
            .map_err(|x| Error::msg(x.to_string()))?;
        Ok(Self { 
            train_img: Array4::from_shape_vec((50000, 3, 32, 32), train_data)?, 
            train_label: Array1::from_shape_vec(50000, train_labels)?, 
            test_img: Array4::from_shape_vec((10000, 3, 32, 32), test_data)?, 
            test_label: Array1::from_shape_vec(10000, test_labels)? 
        })
    }
    pub fn iter_train_img(&self) -> impl Iterator<Item = ArrayView3<u8>> {
        self.train_img.axis_iter(Axis(0))
    }
    pub fn iter_train_label(&self) -> impl Iterator<Item = &u8> {
        self.train_img.iter()
    }
    pub fn iter_test_img(&self) -> impl Iterator<Item = ArrayView3<u8>> {
        self.test_img.axis_iter(Axis(0))
    }
    pub fn iter_test_label(&self) -> impl Iterator<Item = &u8> {
        self.test_label.iter()
    }
}