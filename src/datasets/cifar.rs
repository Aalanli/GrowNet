use std::fs;
use std::path;

use ndarray::prelude::*;

use super::{ImageDataPoint, ImClassifyDataPoint};

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

    pub fn index(&self, indices: &[usize]) -> ImageDataPoint {
        let buf_sz = indices.len() * self.offset;
        let mut buf = Vec::<f32>::new();
        buf.reserve_exact(buf_sz);
        unsafe { buf.set_len(buf_sz); }

        for (i, j) in (0..indices.len()).zip(indices.iter()) {
            let j = *j;
            buf[i*self.offset..(i+1)*self.offset].copy_from_slice(&self.images[j*self.offset..(j+1)*self.offset]);
        }

        let arr = Array::from_shape_vec((indices.len(), self.width, self.height, 3), buf).unwrap();
        ImageDataPoint { image: arr }
    }
}

struct Cifar10 {
    train_images: ContiguousStaticImage,
    test_images: ContiguousStaticImage
}

impl Cifar10 {
    pub fn new(data_folder: &path::Path) -> Self {
        let train_files = (1..5).map(|x| format!("data_batch_{x}.bin"))
            .map(|f| data_folder.join(f)).filter(|f| f.exists());
        let test_file = "test_batch.bin";

        let mut train_images = ContiguousStaticImage::new(32, 32);
        let train_labels = Vec::<u32>::new();
        for file in train_files {
            let raw_buf: Vec<u8> = fs::read(file).unwrap();
            for i in 0..10000 {

            } 
        }
        todo!()
    }

}