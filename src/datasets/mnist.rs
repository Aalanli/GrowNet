use itertools::Itertools;
use ndarray::prelude::*;
use std::fs;
use std::io::{Cursor};
use std::path::PathBuf;
use image::io::Reader as ImageReader;
use rand::thread_rng;
use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};
use bevy_egui::egui;

use super::transforms::{self, Transform};
use super::{DatasetUI, ImClassifyDataPoint, DatasetTypes, Dataset};

/// Main configuration parameters for Mnist
#[derive(Debug, Serialize, Deserialize)]
pub struct MnistParams {
    pub path: String,
    pub batch_size: usize,
    pub transform: MnistTransform
}

impl MnistParams {
    /// Builds a Mnist dataset instance from supplied parameters
    pub fn new(&self) -> Mnist {
        let mut trainset: PathBuf = self.path.clone().into();
        trainset.push("training");
        eprint!("{}", trainset.to_str().unwrap());

        let train_paths = trainset.read_dir().unwrap().map(|path| {
            let a = path.unwrap();
            let label = a.file_name().into_string().unwrap().parse::<u32>().unwrap();
            let subiter = a.path().read_dir().unwrap().map(move |im_file| (label, im_file.unwrap().path()));
            subiter
        }).flatten();

        trainset.pop();
        trainset.push("testing");
        let test_paths = trainset.read_dir().unwrap().map(|path| {
            let a = path.unwrap();
            let label = a.file_name().into_string().unwrap().parse::<u32>().unwrap();
            let subiter = a.path().read_dir().unwrap().map(move |im_file| (label, im_file.unwrap().path()));
            subiter
        }).flatten();

        let mut train = ImClassifyData::new();
        train.push_raw(train_paths);

        let mut test = ImClassifyData::new();
        test.push_raw(test_paths);
        let order = 0..train.labels.len();

        Mnist { train, test, transform: Box::new(self.transform.clone()), order: order.collect_vec(), idx: 0, batch_size: self.batch_size }
    }
}

/// Default is probably not necessary any more, since configuration files exist
/// but we keep them nontheless since there is no default canonical config file
impl Default for MnistParams {
    fn default() -> Self {
        MnistParams { 
            path: "".to_string(), 
            batch_size: 1, 
            transform: MnistTransform {
                transform: transforms::Normalize::default()
            }
        }
    }
}

/// Each dataset supplies its own setup method through its parameters, through egui.
impl DatasetUI for MnistParams {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label("batch size");
                ui.add(egui::DragValue::new(&mut self.batch_size));

                self.transform.ui_setup(ui);
            });
        });
    }
    fn build(&self) -> DatasetTypes {
        DatasetTypes::Classification(Box::new(self.new()))
    }
    fn config(&self) -> String {
        ron::to_string(&self).unwrap()
    }
    fn load_config(&mut self, config: &str) {
        let new_self: Self = ron::from_str(config).unwrap();
        *self = new_self;
    }
}

pub struct Mnist {
    train: ImClassifyData,
    test: ImClassifyData,
    transform: Box<dyn Transform<DataPoint = ImClassifyDataPoint>>,
    order: Vec<usize>,
    idx: usize,
    batch_size: usize
}

/// Just a wrapper for transforms::Normalize, which operates on ImageDataPoints alone
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MnistTransform {
    pub transform: transforms::Normalize
}

impl Transform for MnistTransform {
    type DataPoint = ImClassifyDataPoint;
    fn ui_setup(&mut self, ui: &mut egui::Ui) {
        self.transform.ui_setup(ui);
    }
    fn transform(&self, mut data: Self::DataPoint) -> Self::DataPoint {
        data.image = self.transform.transform(data.image);
        data
    }
}

impl Dataset for Mnist {
    type DataPoint = ImClassifyDataPoint;
    fn next(&mut self) -> Option<Self::DataPoint> {
        // batch size stride greater than the number of elements
        if self.idx + self.batch_size >= self.order.len() {
            return None;
        }
        let img_slice: Vec<_> = (self.idx..self.batch_size).map(|x| {
            &self.train.data[self.order[x]]
        }).collect();
        let img = super::concat_im_size_eq(&img_slice);
        let labels = (self.idx..self.batch_size).map(|x| self.train.labels[self.order[x]]).collect();

        Some(ImClassifyDataPoint { image: img, label: labels })
    }
    fn shuffle(&mut self) {
        self.order.shuffle(&mut rand::thread_rng());
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
            let arr = Array3::from_shape_vec((3, img.width() as usize, img.height() as usize), img.to_vec()).unwrap();
            let arr = arr.mapv(|x| f32::from(x) / 255.0);
            self.data.push(arr);
            self.labels.push(l);
        }
    }
}


#[cfg(test)]
mod test {
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
        let p = h.index_axis(Axis(0), 1);
    }
    
    #[test]
    fn draw() {
        
        use plotters::prelude::*;
        let mut backend = BitMapBackend::new("examples/outputs/2.png", (300, 200));
        // And if we want SVG backend
        // let backend = SVGBackend::new("output.svg", (800, 600));
        //backend.open().unwrap();
        backend.draw_rect((50,50), (200, 150), &RGBColor(255,0,0), true).unwrap();
        //backend.close().unwarp();
        backend.present().unwrap();
    }
}