
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use itertools::Itertools;
use rand::{thread_rng, seq::SliceRandom};
use anyhow::{Error, Result, Context};
use ndarray::prelude::*;

use mnist::*;

macro_rules! shuffle_slice {
    ($slice1:ident$( ,$slices:ident)*) => {
        {
            let len = $slice1.len();
            $(assert!($slices.len() == len);)*
            let mut rng = rand::thread_rng();
    
            for i in 0..len {
                let next = rng.gen_range(i..len);
                if next == i {
                    continue;
                }
        
                unsafe {
                    std::mem::swap(&mut *$slice1.as_mut_ptr().add(i), &mut *$slice1.as_mut_ptr().add(next));
                    $(std::mem::swap(&mut *$slices.as_mut_ptr().add(i), &mut *$slices.as_mut_ptr().add(next));)*
                }
            }
        }
    };
}


pub struct Mnist {
    train_img: Array3<u8>,
    train_label: Array1<u8>,
    train_order: Vec<usize>,
    test_img: Array3<u8>,
    test_label: Array1<u8>,
    test_order: Vec<usize>,
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

        Ok(Mnist { train_img: train_data, train_label: train_labels, test_img: val_data, test_label: val_labels, train_order: (0..60000).collect_vec(), test_order: (0..10000).collect_vec() })
    }

    pub fn iter_train_img(&self) -> impl Iterator<Item = ArrayView2<u8>> {
        self.train_order.iter().map(|x| {
            self.train_img.index_axis(Axis(0), *x)
        })
    }

    pub fn iter_train_label(&self) -> impl Iterator<Item = &u8> {
        self.train_order.iter().map(|x| {
            &self.train_label[*x]
        })
    }

    pub fn iter_test_img(&self) -> impl Iterator<Item = ArrayView2<u8>> {
        self.test_order.iter().map(|x| {
            self.test_img.index_axis(Axis(0), *x)
        })
    }

    pub fn iter_test_label(&self) -> impl Iterator<Item = &u8> {
        self.test_order.iter().map(|x| {
            &self.test_label[*x]
        })
    }
    pub fn shuffle_train(&mut self) {
        let mut rng = thread_rng();
        self.train_order.shuffle(&mut rng);
    }
    pub fn shuffle_test(&mut self) {
        let mut rng = thread_rng();
        self.test_order.shuffle(&mut rng);
    }
}

use rand::Rng;

