use std::mem;
use std::ptr::{self, null};

use ndarray::prelude as np;
use ndarray::{Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};

use rand::{thread_rng};
use rand_distr::Distribution;
use super::GlobalParams;

pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}

type Message = np::Array1<f32>;

/// Each ComputeInstance represents the primary heavy computation
/// performed by each node, this is the most computationally intensive
/// part of a compute node
pub struct ComputeInstance {
    w: np::Array2<f32>,
    b: np::Array1<f32>,
    act_fn: Relu,
    dw: np::Array2<f32>,
    db: np::Array1<f32>,
}

impl ComputeInstance {
    pub fn new(dim: usize) -> ComputeInstance {
        ComputeInstance {
            w:  np::Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b:  np::Array::random((dim,), Uniform::new(-1.0, 1.0)),
            dw: np::Array::zeros((dim,dim)),
            db: np::Array::zeros((dim,)),
            act_fn: Relu{}
        }
    }
    pub fn forward(&mut self, msg: &np::Array1<f32> ) -> np::Array1<f32> {
        // y = W*x + b
        let mut y: np::Array<f32, np::Dim<[usize; 1]>> = self.w.dot(msg);
        y += &self.b;
        // yhat = relu(y)
        y.mapv_inplace(|x| self.act_fn.forward(x));
        y
    }

    pub fn backward(&mut self, grad_msg: &np::Array1<f32>, past_msg: &np::Array1<f32>) -> np::Array1<f32> {
        let dy_dx: np::Array1<f32> = grad_msg.mapv(|x| self.act_fn.backward(x));
        self.db += &dy_dx;
        let past_msg = past_msg.view().insert_axis(Axis(0));
        let dy_dx = dy_dx.view().insert_axis(Axis(1));
        self.dw += &dy_dx.dot(&past_msg);
        let dx_ds = self.dw.t().dot(&dy_dx).remove_axis(Axis(1));
        dx_ds
    }

    pub fn zero_grad(&mut self) {
        self.dw.map_mut(|x| *x = 0.0);
        self.db.map_mut(|x| *x = 0.0);
    }

    pub fn apply_grad(&mut self, params: &GlobalParams) {
        self.w.zip_mut_with(&self.dw, |a, b| *a -= b * params.lr);
        self.b.zip_mut_with(&self.db, |a, b| *a -= b * params.lr);
    }

    pub unsafe fn new_instance(dim: usize) -> *mut Self {
        let compute = ComputeInstance::new(dim);
        let mut compute = mem::ManuallyDrop::new(compute);
        &mut (*compute) as *mut ComputeInstance
    }
}

