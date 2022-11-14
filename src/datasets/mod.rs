use ndarray::prelude::*;
use std::fs;
use std::io::{Cursor};
use std::path::PathBuf;
use image::io::Reader as ImageReader;


pub trait Dataset {
    type Data: Iterator;
    fn train_iter(&self) -> Self::Data;
    fn test_iter(&self) -> Option<Self::Data>;
    fn shuffle(&mut self);
    fn nsamples(&self);
}

struct ImClassifyDataPoint {
    pub image: Array4<f32>,
    pub label: Vec<u16>
}

struct ImClassifyData {
    pub data: Vec<Array3<f32>>,
    pub labels: Vec<u32>,
}

struct ImClassifyIter<'a> {
    batch_size: usize,
    data: &'a ImClassifyData
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

    fn shuffle() {

    }

    fn iter(batch: usize) {

    }

}



struct Mnist {
    train: ImClassifyData,
    test: ImClassifyData,
}

impl Mnist {
    fn new(dir: PathBuf) -> Mnist {
        let mut trainset = dir.clone();
        trainset.push("/training");

        let train_paths = trainset.read_dir().unwrap().map(|path| {
            let a = path.unwrap();
            let label = a.file_name().into_string().unwrap().parse::<u32>().unwrap();
            let subiter = a.path().read_dir().unwrap().map(move |im_file| (label, im_file.unwrap().path()));
            subiter
        }).flatten();

        trainset.pop();
        trainset.push("/testing");
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

        Mnist { train, test }
    }
}

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