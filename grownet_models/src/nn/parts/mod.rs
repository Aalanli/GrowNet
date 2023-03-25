use std::rc::Rc;
use arrayfire::*;
use super::{af_ops, Param};
use af_ops::Float;

use crate::{Flatten, World};

#[derive(Flatten)]
pub struct ConvBlock<T: af_ops::Float> {
    conv: af_ops::conv::Conv2d<T>,
    instance_norm: af_ops::instancenorm::InstanceNorm2D<T>
}

impl<T: af_ops::Float> ConvBlock<T> {
    pub fn new(in_chan: u64, out_chan: u64) -> Self {
        Self { 
            conv: af_ops::conv::Conv2d::new(in_chan, out_chan, [3, 3], [1, 1], [1, 1], false), 
            instance_norm: af_ops::instancenorm::InstanceNorm2D::new(out_chan)
        }
    }

    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl Fn(&mut Self, &Array<T>) -> Array<T>) {
        let (x, f1) = self.conv.forward(x);
        let (x, f2) = self.instance_norm.forward(&x);
        let (x, f3) = af_ops::activations::relu(&x);

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
    max_pool: af_ops::maxpool::MaxPool2D,
    block1: ConvBlock<T>,
    block2: ConvBlock<T>,
}

impl<T: Float> ConvLayer<T> {
    pub fn new(in_chan: u64, out_chan: u64) -> Self {
        Self { 
            pre: ConvBlock::new(in_chan, out_chan), 
            max_pool: af_ops::maxpool::MaxPool2D::new([2, 2], [2, 2]),
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

pub struct SGDSimple<T: Float> {
    pub lr: T,
}

impl<T: Float> SGDSimple<T> {
    pub fn update<'a>(&mut self, world: &mut World<'a>) {
        for param in world.query_mut::<Param<T>>() {
            param.w -= &param.g * self.lr;

        }
        for param in world.query_mut::<Option<Param<T>>>().filter(|x| x.is_some()).map(|x| x.as_mut().unwrap()) {
            param.w -= &param.g * self.lr;
        }
    }
}

pub struct Adam<T: Float> {
    mt_vt: Vec<(Array<T>, Array<T>)>,
    optional_mt_vt: Vec<(Array<T>, Array<T>)>,
    beta1: T,
    beta2: T,
    eps: T,
    t: u64,
}

impl<T: Float> Adam<T> {
    pub fn new<'a>(world: &mut World<'a>, beta1: T, beta2: T) -> Self {
        use af_ops::zeros;
        let mut mt_vt = Vec::new();

        for param in world.query_mut::<Param<T>>() {
            mt_vt.push((zeros(param.dims()), zeros(param.dims())));
        }

        let mut optional_mt_vt = Vec::new();
        for param in world.query_mut::<Option<Param<T>>>() {
            if let Some(param) = param {
                optional_mt_vt.push((zeros(param.dims()), zeros(param.dims())));
            }
        }
        
        Self { mt_vt, optional_mt_vt, beta1, beta2, eps: T::from(1e-6).unwrap(), t: 0 }
    }

    pub fn update_step(param: &mut Param<T>, mt: &mut Array<T>, vt: &mut Array<T>, lr: T, beta1: T, beta2: T, t: u64, eps: T) {
        *mt = &*mt * beta1 + &param.g * (T::one() - beta1);
        *vt = &*vt * beta2 + pow(&*vt, &T::from(2.0).unwrap(), true) * (T::one() - beta2);
        let mhat = &*mt / (T::one() - beta1.powf(T::from(t + 1).unwrap()));
        let vhat = &*vt / (T::one() - beta2.powf(T::from(t + 1).unwrap()));

        param.w -= mhat * lr / (sqrt(&vhat) + eps);
    }

    pub fn update<'a>(&mut self, world: &mut World<'a>, lr: T) {
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        for (param, (mt, vt)) in world.query_mut::<Param<T>>().zip(self.mt_vt.iter_mut()) {
            Self::update_step(param, mt, vt, lr, beta1, beta2, self.t, self.eps);
        }
        for (param, (mt, vt)) in world.query_mut::<Option<Param<T>>>().filter(|x| x.is_some()).zip(self.optional_mt_vt.iter_mut()) {
            let param = param.as_mut().unwrap();
            Self::update_step(param, mt, vt, lr, beta1, beta2, self.t, self.eps);
        }
        self.t += 1;
    }
}


#[test]
fn test_convblock() {
    let x = randn!(28, 28, 3, 1);
    let mut resnet = ConvBlock::new(3, 16);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);
}