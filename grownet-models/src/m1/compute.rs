use std::mem;
use std::ptr::{self, null};

use smallvec::SmallVec;
use ndarray::prelude as np;
use ndarray::{Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};

use rand::{thread_rng};
use rand_distr::Distribution;
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
    temp_var: SmallVec<[np::Array2<f32>; 3]>
}

impl Node {
    fn new(dim: usize) -> Self {
        Self { linear: Linear::new(dim), accum_msg: np::Array2::zeros((1, dim)),
            temp_var: SmallVec::new() }
    }
    fn accum(&mut self, msg: &np::ArrayView2<f32>) {
        if msg.dim() != self.accum_msg.dim() {
            self.accum_msg = np::Array2::zeros(msg.dim());
        }
        self.accum_msg += msg;
    }

    fn forward(&mut self) -> np::Array2<f32> {
        let (batch, hidden) = self.accum_msg.dim();
        

        let arr = self.linear.forward(&self.accum_msg);
        let mut std = np::Array2::<f32>::zeros((1, batch));
        let mut mu = np::Array2::<f32>::zeros((1, batch));
        let mut normed_arr = np::Array2::<f32>::zeros((batch, hidden));

        normed_arr.axis_iter_mut(np::Axis(0)).into_iter()
            .zip(arr.axis_iter(np::Axis(0)))
            .zip(std.iter_mut()).zip(mu.iter_mut())
            .for_each(|(((mut norm, x), std), mu)| {
                let (mu0, std0) = ops::normalize(x.as_slice().unwrap(), norm.as_slice_mut().unwrap(), Some(*mu), Some(*std));
                *mu = mu0;
                *std = std0;
            });
        
        self.temp_var.push(arr);
        self.temp_var.push(mu);
        self.temp_var.push(std);
        normed_arr
    }

    fn accum_backward(&mut self, grad: &np::ArrayView2<f32>) {

    }

    fn backward(&mut self, grad: &np::ArrayView2<f32>) -> np::Array2<f32> {
        let std = self.temp_var.pop().unwrap();
        let mu = self.temp_var.pop().unwrap();
        let arr = self.temp_var.pop().unwrap();
        let shape = grad.dim();
        let mut dy_dx = np::Array2::<f32>::zeros(shape);

        dy_dx.axis_iter_mut(np::Axis(0)).into_iter()
            .zip(arr.axis_iter(np::Axis(0)).into_iter())
            .zip(grad.axis_iter(np::Axis(0)).into_iter())
            .zip(mu.iter())
            .zip(std.iter())
            .for_each(|((((mut dy_dx, x), grad), mu), std)| {
                ops::dnormalize(grad.as_slice().unwrap(), x.as_slice().unwrap(), dy_dx.as_slice_mut().unwrap(), Some(*mu), Some(*std));
            });

        self.linear.backward(&dy_dx, &self.accum_msg)
    }
}

pub struct Grid2D {
    grid: np::Array2<Node>,
    index: [isize; 3],
    width: usize,
    depth: usize,
    d_model: usize
}

impl Grid2D {
    // one-sided padding
    const PAD: usize = 1;
    pub fn new(width: usize, depth: usize, d_model: usize) -> Self {
        let grid = np::Array2::<Node>::from_shape_simple_fn((depth, width + Self::PAD * 2), || {
            Node::new(d_model)
        });
        let mut indices = [0; 3];
        indices[0] = -1;
        indices[1] = 0;
        indices[2] = 1;

        Grid2D { grid, width, depth, d_model, index: indices }
    }

    pub fn forward(&mut self, x: np::Array3<f32>) -> np::Array3<f32> {
        let (b, w, d) = x.dim();
        debug_assert!(w == self.width);
        debug_assert!(d == self.d_model);
        for i in 0..self.width {
            let sliced = x.slice(np::s![.., i, ..]);
            self.grid[(0, i + Self::PAD)].accum(&sliced);
        }

        for i in 0..self.depth - 1 {
            for j in 0..self.width {
                let msg = self.grid[(i, j + Self::PAD)].forward();
                let view = msg.view();
                for k in self.index {
                    self.grid[(i + 1, (Self::PAD as isize + k) as usize)].accum(&view);
                }
            }
        }
        let mut output = np::Array3::<f32>::ones((b, w, d));
        for i in 0..self.width {
            let val = &self.grid[(self.depth - 1, i + Self::PAD)].forward();
            output.slice_mut(np::s![.., i, ..]).zip_mut_with(&val, |x, y| {
                *x = *y;
            });
        }

        output
    }

    pub fn backward(&mut self, grads: np::Array3<f32>) -> np::Array3<f32> {
        todo!()
    }
}

#[test]
fn test_linear() {
    let mut linear = Linear::new(16);
    let input = np::Array2::zeros((14, 16));
    let grad = np::Array2::ones((14, 16));
    
    let y = linear.forward(&input);
    let _dx = linear.backward(&grad, &y);
}

#[test]
fn test_node() {
    let mut node = Node::new(16);
    let input = np::Array2::zeros((14, 16));
    let grad = np::Array2::ones((14, 16));

    node.accum(&input.view());
    let _y = node.forward();
    let _grad = node.backward(&grad.view());
}