
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use anyhow::{Error, Result, Context};
use ndarray::prelude::*;

use mnist::*;

pub struct Mnist {
    train_img: Array3<u8>,
    train_label: Array1<u8>,
    test_img: Array3<u8>,
    test_label: Array1<u8>,
}

impl Mnist {
    pub fn new(base_dir: &str) -> Result<Self> {
        let base_dir = if !base_dir.ends_with("/") {
            base_dir.to_string() + "/"
        } else {
            base_dir.to_string()
        };
        // Deconstruct the returned Mnist struct.
        let mnist::Mnist {
            trn_img,
            trn_lbl,
            tst_img,
        tst_lbl,
            ..
        } = MnistBuilder::new()
            .base_path(&base_dir)
            .download_and_extract()
            .label_format_digit()
            .training_set_length(60_000)
            .validation_set_length(0)
            .test_set_length(10_000)
            .finalize();

        // Can use an Array2 or Array3 here (Array3 for visualization)
        let train_data = Array3::from_shape_vec((60_000, 28, 28), trn_img)
            .context("Error converting images to Array3 struct")?;

        // Convert the returned Mnist struct to Array2 format
        let train_labels = Array1::from_shape_vec(60_000, trn_lbl)
            .context("Error converting training labels to Array2 struct")?;

        let val_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
            .context("Error converting images to Array3 struct")?;

        let val_labels = Array1::from_shape_vec(10_000, tst_lbl)
            .context("Error converting testing labels to Array2 struct")?;

        Ok(Mnist { train_img: train_data, train_label: train_labels, test_img: val_data, test_label: val_labels })
    }

    pub fn iter_train_img(&self) -> impl Iterator<Item = ArrayView2<u8>> {
        self.train_img.axis_iter(Axis(0))
    }
    pub fn iter_train_label(&self) -> impl Iterator<Item = &u8> {
        self.train_img.iter()
    }
    pub fn iter_test_img(&self) -> impl Iterator<Item = ArrayView2<u8>> {
        self.test_img.axis_iter(Axis(0))
    }
    pub fn iter_test_label(&self) -> impl Iterator<Item = &u8> {
        self.test_label.iter()
    }
}
