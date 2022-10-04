use std::{marker::PhantomData, ops::AddAssign};
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};
use num::{self, Float};


struct WeightedSigmoid {
    s: f32,
    b: f32
}

impl WeightedSigmoid {
    pub fn forward(&self, x: f32) -> f32 {
        1.0 / (1.0 + (self.b * (x - self.s)).exp())
    }
}
struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
}

struct Message {
    msg: Array1<f32>,
    mag: f32
}

struct ComputeInstance {
    w: Array2<f32>,
    b: Array1<f32>,
    act_fn: Relu
}

impl ComputeInstance {
    pub fn new(dim: usize) -> ComputeInstance {
        ComputeInstance {
            w: Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b: Array::random((dim,), Uniform::new(-1.0, 1.0)),
            act_fn: Relu{}
        }
    }
    pub fn forward(&self, msg: Message) -> Message {
        // y = W*x + b
        let mut k: Array<f32, Dim<[usize; 1]>> = self.w.dot(&msg.msg);
        k += &self.b;
        // yhat = relu(y)
        k.mapv_inplace(|x| self.act_fn.forward(x));
        let norm = k.fold(0.0, |acc, x| acc + x * x).sqrt();
        Message { msg: k, mag: norm }
    }
}

struct RunningStats {
    window: Vec<f32>,
    moving_avg: f32,
    window_size: u32,
    idx: u32
}

impl RunningStats {
    pub fn new(window: u32) -> Self {
        let mut v = Vec::<f32>::new();
        v.reserve_exact(window as usize);
        RunningStats {
            window: v, moving_avg: 0.0, window_size: window, idx: 0
        }
    }
    pub fn new_stat(&mut self, mag: f32) -> f32 {
        if self.window.len() <= self.window_size as usize {
            self.window.push(mag);
            self.moving_avg += mag;
            return self.moving_avg / self.window.len() as f32;
        } else {
            self.idx %= self.window_size;
            let last_elem = self.window[self.idx as usize];
            self.moving_avg += mag - last_elem;
            self.window[self.idx as usize] = mag;
            self.idx += 1;
            return self.moving_avg / self.window_size as f32;
        }
    }
}

struct ComputeNode {
    act_fn: WeightedSigmoid,
    ro: f32,
    msg_accum: Message,
    stats: RunningStats,
    compute: Box<ComputeInstance>
}

impl ComputeNode {
    pub fn receive(&self, msg: &Message) {

    }
}
