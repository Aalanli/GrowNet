use std::collections::VecDeque;
use std::mem;
use std::ptr;
use ndarray::prelude::*;
use ndarray::{Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};
use num::{self, Float};


struct WeightedSigmoid {
    s: f32,
    b: f32,
    ds: f32,
    db: f32
}

impl WeightedSigmoid {
    pub fn forward(&self, x: f32) -> f32 {
        1.0 / (1.0 + (self.b * (x - self.s)).exp())
    }

    pub fn backward(&mut self, grad: f32, accum_mag: f32, xi: &Array1<f32>) -> Array1<f32> {
        let s = self.forward(accum_mag);
        let d_sigmoid = s * (1.0 - s);
        self.db += -accum_mag * d_sigmoid;
        self.ds += self.b * d_sigmoid;

        let dg_ds = grad * d_sigmoid;
        let dg_dxi = xi * dg_ds / accum_mag;
        dg_dxi
    }

    pub fn zero_grad(&mut self) {
        self.ds = 0.0;
        self.db = 0.0;
    }

    pub fn apply_grad(&mut self, params: GlobalParams) {
        self.b -= self.db * params.lr;
        self.s -= self.ds * params.lr;
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
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
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

impl Message {
    pub fn new(msg: Array1<f32>, sim: Similarity) -> Message {
        let norm = l2_norm(&msg);
        Message { msg, mag: norm, sim }
    }
}

struct ComputeInstance {
    w: Array2<f32>,
    b: Array1<f32>,
    act_fn: Relu,
    dw: Array2<f32>,
    db: Array1<f32>,
}

impl ComputeInstance {
    pub fn new(dim: usize) -> ComputeInstance {
        ComputeInstance {
            w: Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b: Array::random((dim,), Uniform::new(-1.0, 1.0)),
            dw: Array::zeros((dim,dim)),
            db: Array::zeros((dim,)),
            act_fn: Relu{}
        }
    }
    pub fn forward(&mut self, msg: &Array1<f32>) -> Array1<f32> {
        // y = W*x + b
        let mut k: Array<f32, Dim<[usize; 1]>> = self.w.dot(msg);
        k += &self.b;
        // yhat = relu(y)
        k.mapv_inplace(|x| self.act_fn.forward(x));
        k
    }

    pub fn backward(&mut self, grad_msg: &Array1<f32>, past_msg: &Array1<f32>) -> Array1<f32> {
        let dy_dx: Array1<f32> = grad_msg.mapv(|x| self.act_fn.backward(x));
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

    pub fn apply_grad(&mut self, params: GlobalParams) {
        self.w.zip_mut_with(&self.dw, |a, b| *a -= b * params.lr);
        self.b.zip_mut_with(&self.db, |a, b| *a -= b * params.lr);
    }

    pub unsafe fn new_instance(dim: usize) -> *mut Self {
        let compute = ComputeInstance::new(dim);
        let mut compute = mem::ManuallyDrop::new(compute);
        &mut (*compute) as *mut ComputeInstance
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
    compute_initialized: bool,
    past_msgs: VecDeque<(Array1<f32>, f32)>,
}

struct GlobalParams {
    sim_strictness: f32,
    underflow_epsilon: f32,
    sim_epsilon: f32,
    lr: f32,
    compute_dim: usize
}

struct GlobalCoordinates([usize; 2]);

impl ComputeNode {
    pub fn forward(&mut self, msg: &Message, global_params: GlobalParams) -> NodeResult {
        let similarity = Similarity::similarity(&self.sim, &msg.sim, global_params.sim_strictness);
        self.msg_accum.mag += msg.mag * similarity;
        self.msg_accum.msg += &(&msg.msg * similarity);
        if !self.act_fn.is_underflow(self.msg_accum.mag, global_params.underflow_epsilon) {
            let output_vec;
            unsafe {
                if !self.compute_initialized {
                    self.compute = ComputeInstance::new_instance(global_params.compute_dim);
                    self.compute_initialized = true;
                }
                let norm_gate = self.act_fn.forward(self.msg_accum.mag);
                let msg_vec = &self.msg_accum.msg * norm_gate;
                output_vec = (*self.compute).forward(&msg_vec);
                self.past_msgs.push_back((msg_vec, self.msg_accum.mag));
            }
            self.msg_accum.mag = 0.0;
            self.msg_accum.msg.map_inplace(|x| {*x = 0.0;});

            NodeResult::Msg(Message::new(output_vec, self.sim))
        } else {
            NodeResult::NoResult
        }
    }

    pub fn backward(&mut self, grad_msg: &Message, global_params: GlobalParams) -> NodeResult {
        let similarity = Similarity::similarity(&self.sim, &grad_msg.sim, global_params.sim_strictness);
        if similarity <= global_params.sim_epsilon {
            return NodeResult::NoResult;
        }

        let grad = &grad_msg.msg * similarity;
        let (past_msg, accum_mag) = if let Some((x, n)) = self.past_msgs.pop_front() {
            (x, n) } else { panic!("Past undefined"); };
        
        let dy_ds = unsafe { (*self.compute).backward(&grad, &past_msg) };

        let dy_dnorm_gate = dy_ds.sum();
        let mut dy_dxi = self.act_fn.backward(dy_dnorm_gate, accum_mag, &dy_ds);
        let norm_gate = self.act_fn.forward(accum_mag);
        dy_dxi += norm_gate;

        NodeResult::NoResult
    } 

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