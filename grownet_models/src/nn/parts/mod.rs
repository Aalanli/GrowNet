use std::rc::Rc;
use arrayfire::*;
use super::ops;
use ops::Float;

use crate::Flatten;

#[derive(Flatten)]
pub struct ConvBlock<T: ops::Float> {
    conv: ops::conv::Conv2d<T>,
    instance_norm: ops::instancenorm::InstanceNorm2D<T>
}

impl<T: ops::Float> ConvBlock<T> {
    pub fn new(in_chan: u64, out_chan: u64) -> Self {
        Self { 
            conv: ops::conv::Conv2d::new(in_chan, out_chan, [3, 3], [1, 1], [1, 1], false), 
            instance_norm: ops::instancenorm::InstanceNorm2D::new(out_chan)
        }
    }

    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl Fn(&mut Self, &Array<T>) -> Array<T>) {
        let (x, f1) = self.conv.forward(x);
        let (x, f2) = self.instance_norm.forward(&x);
        let (x, f3) = ops::activations::relu(&x);

        let back_fn = move |s: &mut Self, grad: &Array<T>| {
            let g0 = f3(&grad);
            let g1 = f2(&mut s.instance_norm, &g0);
            let g2 = f1(&mut s.conv, &g1);

            g2
        };

        (x, back_fn)
    }
}

#[derive(Flatten)]
pub struct ConvLayer<T: Float> {
    pre: ConvBlock<T>,
    max_pool: ops::maxpool::MaxPool2D,
    block1: ConvBlock<T>,
    block2: ConvBlock<T>,
}

impl<T: Float> ConvLayer<T> {
    pub fn new(in_chan: u64, out_chan: u64) -> Self {
        Self { 
            pre: ConvBlock::new(in_chan, out_chan), 
            max_pool: ops::maxpool::MaxPool2D::new([2, 2], [2, 2]),
            block1: ConvBlock::new(out_chan, out_chan), 
            block2: ConvBlock::new(out_chan, out_chan)
        }
    }

    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl Fn(&mut Self, &Array<T>) -> Array<T>) {
        let (pre, f1) = self.pre.forward(x);
        let (pre, f2)     = self.max_pool.forward::<T>(&pre);
        let (x3,  f3) = self.block1.forward(&pre);
        let (x3,  f4) = self.block2.forward(&x3);

        let y = &pre + &x3;

        let back_fn = move |s: &mut Self, grad: &Array<T>| {
            let dx3 = f4(&mut s.block2, grad);
            let dpre = f3(&mut s.block1, &dx3) + grad;

            let dpre = f2(&dpre);
            let dx = f1(&mut s.pre, &dpre);
            dx
        };

        (y, back_fn)
    }
}


#[test]
fn test_convblock() {
    let x = randn!(28, 28, 3, 1);
    let mut resnet = ConvBlock::new(3, 16);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);
}