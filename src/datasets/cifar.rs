use std::fs;
use std::path;

use ndarray::prelude::*;

use super::Dataset;
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
    pub train_images: ContiguousStaticImage,
    pub train_labels: Vec<u32>,
    pub test_images: ContiguousStaticImage,
    pub test_labels: Vec<u32>,
    pub label_names: Vec<String>
}

impl Cifar10 {
    pub fn new(data_folder: &path::Path) -> Self {
        let train_files = (1..6).map(|x| format!("data_batch_{x}.bin"))
            .map(|f| data_folder.join(f)).filter(|f| f.exists());
        let test_file = "test_batch.bin";

        let num_images = 50000;
        let mut train_images = ContiguousStaticImage::new(32, 32);
        let mut train_labels = Vec::<u32>::new();
        train_labels.reserve_exact(num_images);
        train_images.reserve(num_images);

        let insert_images = |raw_buf: Vec<u8>, labels: &mut Vec<u32>, images: &mut ContiguousStaticImage| {
            for i in 0..10000 {
                let offset = i * 3073;
                labels.push(raw_buf[offset] as u32);
                let iter = (0..1024).map(|j| {
                    let buf_ref = &raw_buf;
                    let temp = (0..3).map(move |k| {
                        buf_ref[offset + 1 + j + k * 1024] as f32
                    });
                    temp  
                }).flatten();
                images.extend_images(iter);
            }
        };
        for file in train_files {
            let raw_buf: Vec<u8> = fs::read(&file).unwrap();
            insert_images(raw_buf, &mut train_labels, &mut train_images);
        }

        let raw_buf: Vec<u8> = fs::read(data_folder.join(test_file)).unwrap();
        let mut test_images = ContiguousStaticImage::new(32, 32);
        let mut test_labels = Vec::<u32>::new();
        test_labels.reserve_exact(10000);
        test_images.reserve(10000);
        insert_images(raw_buf, &mut test_labels, &mut test_images);

        let class_names = fs::read_to_string(data_folder.join("batches.meta.txt")).unwrap();
        let label_names: Vec<String> = class_names.split("\n").map(|x| x.to_string()).collect();

        Cifar10 { train_images, test_images, label_names, train_labels, test_labels }
    }
}

impl Dataset for Cifar10 {
    type DataPoint = ImClassifyDataPoint;

    fn next(&mut self) -> Option<Self::DataPoint> {
        todo!()
    }

    fn reset(&mut self) {
        todo!()
    }

    fn shuffle(&mut self) {
        todo!()
    }
}



#[test]
fn test_cifar10_loading() {
    let path = "assets/ml_datasets/cifar10";
    println!("path exists {}", path::Path::new(path).join("batches.meta.txt").exists());
    let dataset = Cifar10::new(path::Path::new(path));
    let _image = dataset.train_images.index(&[1]);
    println!("num train images {}", dataset.train_images.num_images);
    println!("num test images {}", dataset.test_images.num_images);
    println!("label names {:?}", dataset.label_names);

}