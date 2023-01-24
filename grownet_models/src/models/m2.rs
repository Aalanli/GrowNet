use std::mem;
use std::ops::Deref;
use std::ptr::{self, null};

use ndarray::prelude as np;
use ndarray::prelude::*;
use ndarray::Axis;
use ndarray_rand::{rand_distr::Normal, rand_distr::Uniform, RandomExt};

use num::Float;
use rand::thread_rng;
use rand_distr::Distribution;

pub fn l2_norm<T: Float>(a: &Array1<T>) -> T {
    a.fold(T::zero(), |acc, x| acc + *x * *x).sqrt()
}

/// The global setup parameters for the entire network,
/// is passed by reference into each forward and backward call
#[derive(Clone)]
pub struct GlobalParams {
    sim_strictness: f32,
    underflow_epsilon: f32,
    sim_epsilon: f32,
    lr: f32,
    compute_dim: usize,

    s_mean: f32,
    s_var: f32,
    b_mean: f32,
    b_var: f32,

    ro_var: f32,
}

/// The main enum signifying the output state of a node, as nodes may
/// not output anything if internal requirements are not met
#[derive(Clone)]
pub enum NodeResult<T: Clone> {
    NoResult,
    Msg(T),
}

/// Primary message format in the network, probably should make this
/// polymorphic with respect number of dimensions
#[derive(Clone)]
pub struct Message(Array1<f32>);

/// Primary messages passed between different nodes, should make this a polymorphic
/// trait
#[derive(Clone)]
pub struct NodeMessage {
    msg: Message,
    mag: f32,
}

impl Deref for Message {
    type Target = Array1<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Array1<f32>> for Message {
    fn from(a: Array1<f32>) -> Self {
        Message(a)
    }
}

impl NodeMessage {
    pub fn new(msg: Message) -> NodeMessage {
        let norm = l2_norm(&msg);
        NodeMessage { msg, mag: norm }
    }
}

pub struct WeightedSigmoid {
    pub s: f32,
    pub b: f32,
    pub ds: f32,
    pub db: f32,
    pub x: f32,
    pub y: f32,
}

impl WeightedSigmoid {
    pub fn forward(&mut self, x: f32) -> f32 {
        self.x = x;
        self.y = 1.0 / (1.0 + (self.b * (x - self.s)).exp());
        self.y
    }

    pub fn backward(&mut self, grad: f32) -> f32 {
        let dy_dsigmoid = grad * self.y * (1.0 - self.y);
        let dy_dx = self.b * dy_dsigmoid;
        self.ds += -dy_dx;
        self.db += dy_dsigmoid * (self.x - self.s);
        dy_dx
    }

    pub fn zero_grad(&mut self) {
        self.ds = 0.0;
        self.db = 0.0;
    }

    pub fn apply_grad(&mut self, params: &GlobalParams) {
        self.b -= self.db * params.lr;
        self.s -= self.ds * params.lr;
    }

    pub fn is_underflow(&self, x: f32, eps: f32) -> bool {
        self.s - eps / self.b > x
    }

    pub fn underflow_cutoff(&self, eps: f32) -> f32 {
        self.s - eps / self.b
    }
}
pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}

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
            w: np::Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b: np::Array::random((dim,), Uniform::new(-1.0, 1.0)),
            dw: np::Array::zeros((dim, dim)),
            db: np::Array::zeros((dim,)),
            act_fn: Relu {},
        }
    }
    pub fn forward(&mut self, msg: &Message) -> np::Array1<f32> {
        // y = W*x + b
        let mut y: np::Array<f32, np::Dim<[usize; 1]>> = self.w.dot(&**msg);
        y += &self.b;
        // yhat = relu(y)
        y.mapv_inplace(|x| self.act_fn.forward(x));
        y
    }

    pub fn backward(
        &mut self,
        grad_msg: &np::Array1<f32>,
        past_msg: &np::Array1<f32>,
    ) -> np::Array1<f32> {
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

/// Each ComputeNode encapsulates the state of each embedded node in the
/// computation matrix. Analogy to brain cells. Like in neurons, if an
/// activation requirement is not met, then no action occurs, as represented
/// here.
pub struct ComputeNode {
    pub act_fn: WeightedSigmoid,
    pub msg_accum: Message,
    pub mag: f32,
    compute: *mut ComputeInstance,
    compute_initialized: bool,
}

impl ComputeNode {
    pub fn new(params: &GlobalParams) -> ComputeNode {
        let mut rng = thread_rng();
        let mut norm = |mu, si| Normal::new(mu, si).unwrap().sample(&mut rng);
        let act_fn = WeightedSigmoid {
            s: norm(params.s_mean, params.s_var),
            b: norm(params.b_mean, params.b_var),
            ds: 0.0,
            db: 0.0,
            x: 0.0,
            y: 0.0,
        };
        let msg_accum: np::Array1<f32> = np::Array1::zeros((params.compute_dim,));

        ComputeNode {
            act_fn,
            msg_accum: msg_accum.into(),
            mag: 0.0,
            compute: null::<ComputeInstance>() as *mut ComputeInstance,
            compute_initialized: false,
        }
    }

    // if the node becomes inactive, then drop its memory
    // consider setting up an arena allocator for the compute units
    fn drop_compute(&mut self) {
        unsafe {
            if self.compute_initialized {
                ptr::drop_in_place(self.compute);
            }
        }
        self.compute_initialized = false;
    }

    fn inner_compute_forward(&mut self, msg: &Message, global_params: &GlobalParams) -> Message {
        unsafe {
            if !self.compute_initialized {
                self.compute = ComputeInstance::new_instance(global_params.compute_dim);
                self.compute_initialized = true;
            }
            return (*self.compute).forward(&*msg).into();
        }
    }

    fn inner_compute_backward(&mut self, grad: &Message) -> Message {
        unsafe {
            if !self.compute_initialized {
                let temp = grad.clone();
                temp
            } else {
                (*self.compute).backward(grad, &self.msg_accum).into()
            }
        }
    }

    pub fn forward(
        &mut self,
        msg: &NodeMessage,
        global_params: &GlobalParams,
    ) -> NodeResult<NodeMessage> {
        self.msg_accum.0 += &*msg.msg;
        self.mag = l2_norm(&self.msg_accum.0);

        if !self
            .act_fn
            .is_underflow(self.mag, global_params.underflow_epsilon)
        {
            let norm_gate = self.act_fn.forward(self.mag);
            let scaled_msg = &self.msg_accum.0 * norm_gate;
            let out_msg = self.inner_compute_forward(&scaled_msg.into(), &global_params);

            NodeResult::Msg(NodeMessage {
                msg: out_msg,
                mag: self.mag,
            })
        } else {
            NodeResult::NoResult
        }
    }

    pub fn backward(
        &mut self,
        grad_msg: &Message,
        _global_params: &GlobalParams,
    ) -> NodeResult<NodeMessage> {
        let mut dl_dyi = self.inner_compute_backward(grad_msg);
        let dy_dw = dl_dyi.0.dot(&*self.msg_accum);
        let dw_dt = self.act_fn.backward(dy_dw);
        dl_dyi.0.zip_mut_with(&self.msg_accum.0, |dl_dxi, xi| {
            *dl_dxi = *dl_dxi * self.act_fn.y + dw_dt * xi / self.mag;
        });

        let grad_norm = l2_norm(&dl_dyi);
        NodeResult::Msg(NodeMessage {
            msg: dl_dyi,
            mag: grad_norm,
        })
    }

    pub fn apply_grad(&mut self, global_params: &GlobalParams) {
        self.act_fn.apply_grad(global_params);
        unsafe {
            if self.compute_initialized {
                (*self.compute).apply_grad(global_params);
            }
        }
    }

    pub fn zero_grad(&mut self) {
        self.act_fn.zero_grad();
        unsafe {
            if self.compute_initialized {
                (*self.compute).zero_grad();
            }
        }
    }

    pub fn zero_accum(&mut self) {
        self.msg_accum.0.mapv_inplace(|_| 0.0);
        self.mag = 0.0
    }
}

impl Drop for ComputeNode {
    fn drop(&mut self) {
        self.drop_compute();
    }
}
