use std::marker::PhantomData;

use anyhow::{Result, Error};
use arrayfire::*;
use crossbeam::channel::unbounded;

use crate::nn::ops::{self as ops, *};
use crate::nn::parts::*;
use crate::datasets::{transforms, mnist};

use crate::{Flatten, World, Config, config, Options, opt};
use crate::nn::parts::Adam;

#[derive(Flatten)]
 pub struct FastResnet<T: Float> {
    pre: ConvBlock<T>,
    layer1: ConvLayer<T>,
    inter: ConvBlock<T>,
    max_pool: ops::maxpool::MaxPool2D,
    layer2: ConvLayer<T>,
    max_pool2: ops::maxpool::MaxPool2D,
    linear: ops::linear::Linear<T>
}

impl<T: Float> FastResnet<T> {
    pub fn new(classes: u64) -> Self {
        Self { 
            pre: ConvBlock::new(3, 64), 
            layer1: ConvLayer::new(64, 128), 
            inter: ConvBlock::new(128, 256), 
            max_pool: ops::maxpool::MaxPool2D::new([2, 2], [2, 2]), 
            layer2: ConvLayer::new(256, 512), 
            max_pool2: ops::maxpool::MaxPool2D::new([3, 3], [2, 2]), 
            linear: ops::linear::Linear::new(512, classes, true)
        }
    }

    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl Fn(&mut Self, &Array<T>) -> Array<T>) {
        let (x1, df1) = self.pre.forward(x);
        let (x2, df2) = self.layer1.forward(&x1);
        let (x3, df3) = self.inter.forward(&x2);
        let (x4, df4) = self.max_pool.forward(&x3);
        let (x5, df5) = self.layer2.forward(&x4);
        
        let (x6, df6) = self.max_pool2.forward(&x5); // [w, h, c, b]
        let dims = x6.dims();
        let (x7, df7) = ops::reshape(&x6, dim4!(dims[0] * dims[1], dims[2], dims[3]));
        let (x8, df8) = ops::reduce_sum(&x7, 0);
        let (x9, df9) = ops::reshape(&x8, dim4!(dims[2], dims[3]));
        let (x10, df10) = self.linear.forward(&x9);

        let df = move |s: &mut Self, grad: &Array<T>| {
            let dx9 = df10(&mut s.linear, grad);
            let dx8 = df9(&dx9);
            let dx7 = df8(&dx8);
            let dx6 = df7(&dx7);
            let dx5 = df6(&dx6);
            let dx4 = df5(&mut s.layer2, &dx5);
            let dx3 = df4(&dx4);
            let dx2 = df3(&mut s.inter, &dx3);
            let dx1 = df2(&mut s.layer1, &dx2);
            let dx = df1(&mut s.pre, &dx1);
            dx
        };

        (x10, df)
    }
}

pub fn baseline_config() -> Config {
    config!(
        ("lr", 0.008),
        ("batch_size", 8),
        ("epochs", 10)
    )
}

use ndarray as nd;
use image;
use itertools::Itertools;

use super::TrainProcess;
fn transform_data<'a>(imgs: impl Iterator<Item = nd::ArrayView2<'a, u8>> + 'a, batch_size: usize) -> impl Iterator<Item = nd::Array4<f32>> + 'a {
    let pre_iter = imgs
        .map(|bk_img| {
            let bk_img = bk_img.to_owned();
            let im = transforms::to_image_grayscale(bk_img);
            let rgb_im = image::DynamicImage::ImageLuma8(im).to_rgb8();
            let array = transforms::from_image(rgb_im, false);
            array.map(|x| *x as f32 / 255.0)
        });
    Batcher::new(pre_iter, batch_size).map(|x| {
        transforms::batch_im(&x)
    })
}

fn accuracy(logits: &Array<f32>, labels: &Array<u32>) -> f32 {
    let (_, index) = af::imax(logits, 0);
    let avg = af::mean(&af::eq(&index, &moddims(&labels, dim4!(1, labels.dims()[0])), false), 1);
    let mut acc = [0.0f32];
    avg.host(&mut acc);
    acc[0]
}

pub struct Batcher<T> {
    iter: T,
    len: usize,
}

impl<T> Batcher<T> {
    pub fn new(iter: T, len: usize) -> Self {
        Self { iter, len }
    }
}

impl<T, Item> Iterator for Batcher<T>
where T: Iterator<Item = Item> {
    type Item = Vec<Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut span = Vec::new();
        span.reserve_exact(self.len);
        for _ in 0..self.len {
            if let Some(x) = self.iter.next() {
                span.push(x);
            } else {
                return None;
            }
        }
        Some(span)
    }
}

pub trait Batch<T: Iterator> {
    fn batch(self, batch_size: usize) -> Batcher<T>;
}

impl<T, Item> Batch<T> for T
where T: Iterator<Item = Item> {
    fn batch(self, batch_size: usize) -> Batcher<T> {
        Batcher::new(self, batch_size)
    }
}

pub fn run(config: &Config) -> Result<TrainProcess> {
    use super::{PlotPoint, TrainRecv, TrainSend, RunStats};
    let lr: f64 = config.uget("lr").into();
    let batch_size: isize = config.uget("batch_size").into();
    let epochs: isize = config.uget("epochs").into();

    let (command_sender, command_recv) = unbounded::<TrainSend>();
    let (log_sender, log_recv) = unbounded::<TrainRecv>();

    let train_log_steps: isize = config.uget("train_log_steps").into();
    let data_dir: String = config.uget("dataset_path").into();
    let dataset = mnist::Mnist::new(&data_dir)?;

    let sender = log_sender;
    let recv = command_recv;
    let handle = std::thread::spawn(move || {
        let dataset = dataset;
        
        let mut model = FastResnet::<f32>::new(10);
        let mut adam = {
            let mut world = World::new();
            model.flatten("".to_string(), &mut world);
            
            world = world.clear();
            Adam::new(&mut world, 0.88, 0.99) 
        };

        let mut steps = 0;
        let mut running_loss = 0.0;
        let mut running_acc = 0.0;
        let mut steps_since_last_log = 0;

        let setup_test_iter = || {
            let test_iter = dataset.iter_test_img();
            let test_imgs = transform_data(test_iter, batch_size as usize);
            let test_labels = dataset.iter_test_label().map(|x| *x)
                .batch(batch_size as usize)
                .map(|x| { nd::Array1::from_vec(x) });
            let test_combined = test_imgs.zip(test_labels).map(|(img, label)| {
                (transforms::to_afarray(&img), Array::new(label.as_slice().unwrap(), dim4!(label.len() as u64)))
            });
            test_combined
        };

        let mut test_iter = setup_test_iter();

        for epoch in 0..epochs {
            let train_iter = dataset.iter_train_img();
            let train_imgs = transform_data(train_iter, batch_size as usize);
            let train_labels = dataset.iter_train_label().map(|x| *x)
                .batch(batch_size as usize)
                .map(|x| { nd::Array1::from_vec(x) });
            let train_iter = train_imgs.zip(train_labels).map(|(img, label)| {
                (transforms::to_afarray(&img), Array::new(label.as_slice().unwrap(), dim4!(label.len() as u64)))
            });

            for (img, label) in train_iter {
                steps += 1isize;
                let (logits, df) = model.forward(&img);
                let (loss, dl_dlogit) = ops::loss::cross_entropy(&logits, &ops::loss::one_hot(label.cast(), 10));
                let dl = dl_dlogit(&Array::new(&[1.0], dim4!(1)));
                df(&mut model, &dl);

                let mut world = World::new();
                model.flatten("".to_string(), &mut world);
                adam.update(&mut world, lr);

                let mut loss_host = [0.0f32];
                loss.host(loss_host.as_mut_slice());

                running_loss += loss_host[0];
                running_acc += accuracy(&logits, &label.cast());
                steps_since_last_log += 1isize;

                if steps % train_log_steps == 0 {
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train loss", 
                            x_title: "step", 
                            y_title: "cross entropy", 
                            x: steps as f64, 
                            y: (running_loss / steps_since_last_log as f32) as f64
                        }))
                        .unwrap();
                    sender
                        .send(TrainRecv::PLOT(super::PlotPoint { 
                            title: "train accuracy", 
                            x_title: "step", 
                            y_title: "accuracy", 
                            x: steps as f64, 
                            y: (running_acc / steps_since_last_log as f32) as f64
                        })).unwrap();
                    steps_since_last_log = 1;
                    running_acc = 0.0;
                    running_loss = 0.0;
                }

                if let Ok(TrainSend::KILL) = recv.try_recv() {
                    return;
                }
            }

        }
    });
    Ok(TrainProcess {
        send: command_sender,
        recv: log_recv,
        handle: Some(handle),
    })
}

#[test]
fn test_fastresnet() {
    let x = randn!(28, 28, 3, 8);
    let mut resnet = FastResnet::new(10);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);

    use crate::World;
    let mut world = World::from(&mut resnet);
    for (path, item) in world.query_mut_with_path::<Param<f32>>() {
        println!("{}, params {}", path, item.w.elements());
    }  
}

#[test]
fn bench_fastresnet() {
    let x = randn!(128, 128, 3, 8);
    let mut resnet = FastResnet::new(10);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);

    use std::time::Instant;
    let inst = Instant::now();
    for _ in 0..10 {
        let (y, df) = resnet.forward(&x);
        let _grad = df(&mut resnet, &y);
        _grad.eval();
    }

    println!("avg {} sec/step", inst.elapsed().as_secs_f32() / 10.0);
}


#[test]
fn test_adam_update() {
    let x = randn!(128, 128, 3, 8);
    let mut resnet = FastResnet::<f32>::new(10);
    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);

    let mut world = World::new();
    resnet.flatten("".to_string(), &mut world);
    let mut adam = Adam::new(&mut world, 0.8f32, 0.999f32);

    adam.update(&mut world, 0.02);
}