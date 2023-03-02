use std::rc::Rc;
use arrayfire::*;
use super::ops;
use ops::Float;

pub struct ConvBlock<T: ops::Float> {
    conv: ops::conv::Conv2d<T>,
    bnorm: ops::batchnorm2d::BatchNorm2D<T>
}

impl<T: ops::Float> ConvBlock<T> {
    pub fn new(in_chan: u64, out_chan: u64) -> Self {
        Self { 
            conv: ops::conv::Conv2d::new(in_chan, out_chan, [3, 3], [1, 1], [1, 1], false), 
            bnorm: ops::batchnorm2d::BatchNorm2D::new(out_chan)
        }
    }

    pub fn forward(&mut self, x: &Rc<Array<T>>) -> (Rc<Array<T>>, impl FnMut(&mut Self, &Array<T>) -> Array<T>) {
        let (x, mut f1) = self.conv.forward(x);
        let (x, mut f2) = self.bnorm.forward(&x);

        let back_fn = move |s: &mut Self, grad: &Array<T>| {
            let g1 = f2(&mut s.bnorm, grad);
            let g2 = f1(&mut s.conv, &g1);

            g2
        };

        (x, back_fn)
    }
}

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

    pub fn forward(&mut self, x: &Rc<Array<T>>) -> (Rc<Array<T>>, impl FnMut(&mut Self, &Array<T>) -> Array<T>) {
        let (pre, mut f1) = self.pre.forward(x);
        let (pre, mut f2) = self.max_pool.forward::<T>(&pre);
        let (x3,  mut f3) = self.block1.forward(&pre);
        let (x3,  mut f4) = self.block2.forward(&x3);

        let y = &*pre + &* x3;

        let back_fn = move |s: &mut Self, grad: &Array<T>| {
            let dx3 = f4(&mut s.block2, grad);
            let dpre = f3(&mut s.block1, &dx3) + grad;

            let dpre = f2(&mut s.max_pool, &dpre);
            let dx = f1(&mut s.pre, &dpre);
            dx
        };

        (Rc::new(y), back_fn)
    }
}

pub struct FastResnet<T: Float> {
    pre: ConvLayer<T>,
}
