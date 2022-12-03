use std::mem;
use std::ptr::{self, null};

use ndarray::prelude as np;
use ndarray::{Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};

use rand::{thread_rng};
use rand_distr::Distribution;
use super::GlobalParams;
use crate::ops;

pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}

pub struct Linear {
    w: np::Array2<f32>,
    b: np::Array2<f32>,
    act_fn: Relu,
    dw: np::Array2<f32>,
    db: np::Array2<f32>,
}

impl Linear {
    pub fn new(dim: usize) -> Linear {
        Linear {
            w:  np::Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b:  np::Array::zeros((1, dim)),
            dw: np::Array::zeros((dim,dim)),
            db: np::Array::zeros((1, dim)),
            act_fn: Relu{}
        }
    }
    
    pub fn forward(&mut self, msg: &np::Array2<f32>) -> np::Array2<f32> {
        // y = W*x + b
        let mut y = msg.dot(&self.w);
        y += &self.b;
        // yhat = relu(y)
        y.mapv_inplace(|x| self.act_fn.forward(x));
        y
    }

    pub fn backward(&mut self, grad_msg: &np::Array2<f32>, input_msg: &np::Array2<f32>) -> np::Array2<f32> {
        let dy_dx = grad_msg.mapv(|x| self.act_fn.backward(x));
        self.db += &dy_dx.sum_axis(np::Axis(0));
        self.dw += &input_msg.t().dot(&dy_dx);
        let dx_ds = dy_dx.dot(&self.w.t());
        dx_ds
    }

    pub fn get_parms<'a, 'b: 'a>(&'b mut self, params: &'a mut Vec<(&'a mut np::Array2<f32>, &'a mut np::Array2<f32>)>) {
        params.push((&mut self.w, &mut self.dw));
        params.push((&mut self.b, &mut self.db));
    }
}

struct Node {
    linear: Linear,
    accum_msg: np::Array2<f32>,
    norm_msg: np::Array2<f32>,
}

impl Node {
    fn new(dim: usize) -> Self {
        Self { linear: Linear::new(dim), accum_msg: np::Array2::zeros((1, dim)),
            norm_msg: np::Array2::zeros((1, dim)) }
    }
    fn accum(&mut self, msg: &np::Array2<f32>) {
        if msg.dim() != self.accum_msg.dim() {
            self.accum_msg = np::Array2::zeros(msg.dim());
        }
        self.accum_msg += msg;
    }

    fn forward(&mut self) -> np::Array2<f32> {
        let batch = self.accum_msg.dim().0;
        
        self.linear.forward(&self.accum_msg)
    }

    fn backward(&mut self, grad: &np::Array2<f32>) {
        self.linear.backward(grad, &self.accum_msg);
    }
}

#[test]
fn linear() {
    let mut linear = Linear::new(16);
    let input = np::Array2::zeros((14, 16));
    let grad = np::Array2::ones((14, 16));
    
    let y = linear.forward(&input);
    let dx = linear.backward(&grad, &y);
}