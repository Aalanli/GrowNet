use std::mem;
use std::ptr;
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

    pub fn is_underflow(&self, x: f32, eps: f32) -> bool {
        self.s - eps / self.b > x
    }
}
struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
}

#[derive(Clone, Copy)]
struct Similarity {
    ro: f32
}

impl Similarity {
    fn similarity(a: &Similarity, b: &Similarity, strictness: f32) -> f32 {
        strictness * ((a.ro - b.ro).cos() - 1.0).exp()
    }
}

struct Message {
    msg: Array1<f32>,
    mag: f32,
    sim: Similarity
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
    pub fn forward(&self, msg: &Message) -> Message {
        // y = W*x + b
        let mut k: Array<f32, Dim<[usize; 1]>> = self.w.dot(&msg.msg);
        k += &self.b;
        // yhat = relu(y)
        k.mapv_inplace(|x| self.act_fn.forward(x));
        let norm = l2_norm(&k);
        Message { msg: k, mag: norm, sim: msg.sim }
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

enum NodeResult {
    NoResult,
    Msg(Message),
}


struct ComputeNode {
    act_fn: WeightedSigmoid,
    sim: Similarity,
    msg_accum: Message,
    compute: *mut ComputeInstance,
    compute_dim: usize,
    compute_initialized: bool,
}

struct GlobalParams {
    sim_strictness: f32,
    underflow_epsilon: f32
}

struct GlobalCoordinates([usize; 2]);

impl ComputeNode {
    pub fn forward(&mut self, msg: &Message, global_params: GlobalParams) -> NodeResult {
        let similarity = Similarity::similarity(&self.sim, &msg.sim, global_params.sim_strictness);
        self.msg_accum.mag += msg.mag * similarity;
        self.msg_accum.msg += &(&msg.msg * similarity);
        if !self.act_fn.is_underflow(self.msg_accum.mag, global_params.underflow_epsilon) {
            let output_msg;
            unsafe {
                if !self.compute_initialized {
                    let compute = ComputeInstance::new(self.compute_dim);
                    let mut compute = mem::ManuallyDrop::new(compute);
                    self.compute = &mut (*compute) as *mut ComputeInstance;
                    self.compute_initialized = true;
                }
                output_msg = (*self.compute).forward(&self.msg_accum);    
            }
            self.msg_accum.mag = 0.0;
            self.msg_accum.msg *= 0.0;
            NodeResult::Msg(output_msg)
        } else {
            NodeResult::NoResult
        }
    }

    pub fn backward() {} 

    pub fn forward_nodes() {}

    pub fn backward_nodes() {}
}

impl Drop for ComputeNode {
    fn drop(&mut self) {
        unsafe {
            if self.compute_initialized {
                ptr::drop_in_place(self.compute);
            }
        }
    }
}

fn l2_norm<T: Float>(a: &Array1<T>) -> T {
    a.fold(T::zero(), |acc, x| acc + *x * *x).sqrt()
}

#[cfg(test)]
mod test {
    #[test]
    fn underflow() {
        for i in 0..88 {
            let x = i as f32;
            let y = 1.0 / (1.0 + x.exp());
            if !(y > 0.0) {
                panic!("overflow at {i}");
            }
        }
    }
}