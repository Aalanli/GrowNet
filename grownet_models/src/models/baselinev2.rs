use crate::nn::ops::{self as ops, *};
use arrayfire::*;
use crate::nn::parts::*;

use crate::{Flatten, World};

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