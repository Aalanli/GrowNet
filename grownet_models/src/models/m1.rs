use std::mem;
use std::ptr::{self, null};

use ndarray::Axis;
use ndarray::{prelude as np, Array2, Dimension};
use ndarray_rand::{rand_distr::Normal, rand_distr::Uniform, RandomExt};
use num::Float;
use smallvec::SmallVec;

use crate::ops;
use rand::thread_rng;
use rand_distr::Distribution;

pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}

pub struct GradientParams<'a> {
    pub param: &'a mut [f32],
    grad: &'a mut [f32],
}

impl<'a> GradientParams<'a> {
    pub fn new<D: Dimension>(
        param: &'a mut np::Array<f32, D>,
        grad: &'a mut np::Array<f32, D>,
    ) -> Self {
        Self {
            param: param.as_slice_mut().unwrap(),
            grad: grad.as_slice_mut().unwrap(),
        }
    }
}

type Grads<'a> = Vec<GradientParams<'a>>;

pub struct GridStats {
    node_act: Array2<f32>,
}

pub struct Adam {
    mt: Vec<f32>,
    vt: Vec<f32>,
    alpha: f32,
    beta1: f32,
    beta2: f32,
    tracked_sizes: Vec<usize>,
}

impl Adam {
    const EPS: f32 = 1e-8;
    fn new(alpha: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            tracked_sizes: Vec::new(),
            mt: Vec::new(),
            vt: Vec::new(),
            alpha,
            beta1,
            beta2,
        }
    }

    fn init<'a>(&mut self, grads: Grads<'a>) {
        self.tracked_sizes.reserve_exact(grads.len());
        let mut offset = 0;
        for gradpm in &grads {
            debug_assert!(gradpm.param.len() == gradpm.grad.len());
            self.tracked_sizes.push(gradpm.param.len());
            offset += gradpm.param.len();
        }
        self.mt = vec![0.0; offset];
        self.vt = vec![0.0; offset];
    }

    fn apply_grad<'a>(&mut self, mut grads: Grads<'a>) {
        let mut offset = 0;
        for (i, gradpm) in grads.iter_mut().enumerate() {
            let len = gradpm.grad.len();
            debug_assert!(gradpm.param.len() == gradpm.grad.len());
            debug_assert!(gradpm.param.len() == self.tracked_sizes[i]);
            let mt = &mut self.mt[offset..offset + len];
            let vt = &mut self.vt[offset..offset + len];
            mt.iter_mut()
                .zip(vt.iter_mut())
                .zip(gradpm.grad.iter_mut())
                .zip(gradpm.param.iter_mut())
                .for_each(|(((m, v), g), x)| {
                    *m = self.beta1 * *m + (1.0 - self.beta1) * *g;
                    *v = self.beta2 * *v + (1.0 - self.beta2) * (*g).powi(2);
                    let alpha = self.alpha * (1.0 - self.beta2).sqrt() / (1.0 - self.beta1);
                    *x = *x - alpha * *m / ((*v).sqrt() + Self::EPS);
                    *g = 0.0;
                });
            offset += len;
        }
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
            w: np::Array::random((dim, dim), Normal::new(0.0, 1.0).unwrap()),
            b: np::Array::zeros((1, dim)),
            dw: np::Array::zeros((dim, dim)),
            db: np::Array::zeros((1, dim)),
            act_fn: Relu {},
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

    pub fn backward(
        &mut self,
        grad_msg: &np::Array2<f32>,
        input_msg: &np::Array2<f32>,
    ) -> np::Array2<f32> {
        let dy_dx = grad_msg.mapv(|x| self.act_fn.backward(x));
        self.db += &dy_dx.sum_axis(np::Axis(0));
        self.dw += &input_msg.t().dot(&dy_dx);
        let dx_ds = dy_dx.dot(&self.w.t());
        dx_ds
    }

    pub fn get_grads<'a>(&'a mut self, grads: &mut Vec<GradientParams<'a>>) {
        grads.push(GradientParams::new(&mut self.w, &mut self.dw));
        grads.push(GradientParams::new(&mut self.b, &mut self.db));
    }
}

struct Node {
    linear: Linear,
    accum_msg: np::Array2<f32>,
    grad_accum: np::Array2<f32>,
    temp_var: SmallVec<[np::Array2<f32>; 3]>,
}

impl Node {
    fn new(dim: usize) -> Self {
        Self {
            linear: Linear::new(dim),
            accum_msg: np::Array2::zeros((1, dim)),
            grad_accum: np::Array2::zeros((1, dim)),
            temp_var: SmallVec::new(),
        }
    }

    fn accum(&mut self, msg: &np::ArrayView2<f32>) {
        if msg.dim() != self.accum_msg.dim() {
            self.accum_msg = np::Array2::zeros(msg.dim());
        }
        self.accum_msg += msg;
    }

    fn accum_backward(&mut self, grad: &np::ArrayView2<f32>) {
        if grad.dim() != self.grad_accum.dim() {
            self.grad_accum = np::Array2::zeros(grad.dim());
        }
        self.grad_accum += grad;
    }

    fn reset(&mut self) {
        self.temp_var.clear();
        self.accum_msg.map_inplace(|x| *x = 0.0);
        self.grad_accum.map_inplace(|x| *x = 0.0);
    }

    fn forward(&mut self) -> np::Array2<f32> {
        let (batch, hidden) = self.accum_msg.dim();

        let arr = self.linear.forward(&self.accum_msg);
        let mut std = np::Array2::<f32>::zeros((1, batch));
        let mut mu = np::Array2::<f32>::zeros((1, batch));
        // the buffer containing the normalized version of arr
        let mut normed_arr = np::Array2::<f32>::zeros((batch, hidden));
        // normalize on each row of arr, storing the mean and standard deviation for efficiency
        normed_arr
            .axis_iter_mut(np::Axis(0))
            .into_iter()
            .zip(arr.axis_iter(np::Axis(0)))
            .zip(std.iter_mut())
            .zip(mu.iter_mut())
            .for_each(|(((mut norm, x), std), mu)| {
                let (mu0, std0) = ops::normalize(
                    x.as_slice().unwrap(),
                    norm.as_slice_mut().unwrap(),
                    None,
                    None,
                );
                *mu = mu0;
                *std = std0;
            });

        // zero the accum buffer
        self.accum_msg.map_inplace(|x| {
            *x = 0.0;
        });

        self.temp_var.push(arr);
        self.temp_var.push(mu);
        self.temp_var.push(std);
        normed_arr
    }

    fn backward(&mut self) -> np::Array2<f32> {
        let grad = self.grad_accum.view();
        let std = self.temp_var.pop().unwrap();
        let mu = self.temp_var.pop().unwrap();
        let arr = self.temp_var.pop().unwrap();
        let shape = grad.dim();
        let mut dy_dx = np::Array2::<f32>::zeros(shape);

        dy_dx
            .axis_iter_mut(np::Axis(0))
            .into_iter()
            .zip(arr.axis_iter(np::Axis(0)).into_iter())
            .zip(grad.axis_iter(np::Axis(0)).into_iter())
            .zip(mu.iter())
            .zip(std.iter())
            .for_each(|((((mut dy_dx, x), grad), mu), std)| {
                ops::dnormalize(
                    grad.as_slice().unwrap(),
                    x.as_slice().unwrap(),
                    dy_dx.as_slice_mut().unwrap(),
                    Some(*mu),
                    Some(*std),
                );
            });

        self.grad_accum.map_inplace(|x| {
            *x = 0.0;
        });

        self.linear.backward(&dy_dx, &self.accum_msg)
    }

    fn get_grads<'a>(&'a mut self, grads: &mut Vec<GradientParams<'a>>) {
        self.linear.get_grads(grads);
    }
}

pub struct Grid2D {
    grid: np::Array2<Node>,
    index: [isize; 3],
    width: usize,
    depth: usize,
    d_model: usize,
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

        Grid2D {
            grid,
            width,
            depth,
            d_model,
            index: indices,
        }
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
                //println!("layer {} node {}, max val {}", i, j, maximum(&view));
                for k in self.index {
                    self.grid[(i + 1, ((j + Self::PAD) as isize + k) as usize)].accum(&view);
                }
            }
        }
        let mut output = np::Array3::<f32>::ones((b, w, d));
        for i in 0..self.width {
            let val = &self.grid[(self.depth - 1, i + Self::PAD)].forward();
            output
                .slice_mut(np::s![.., i, ..])
                .zip_mut_with(&val, |x, y| {
                    *x = *y;
                });
        }

        output
    }

    /// In the case of this network, the backwards pass is fairly isomorphic to the forward pass
    pub fn backward(&mut self, grad: np::Array3<f32>) -> np::Array3<f32> {
        let (b, w, d) = grad.dim();
        debug_assert!(w == self.width);
        debug_assert!(d == self.d_model);
        for i in 0..self.width {
            let sliced = grad.slice(np::s![.., i, ..]);
            self.grid[(0, i + Self::PAD)].accum_backward(&sliced);
        }

        for i in 0..self.depth - 1 {
            for j in 0..self.width {
                let msg = self.grid[(i, j + Self::PAD)].backward();
                let view = msg.view();
                for k in self.index {
                    self.grid[(i + 1, ((j + Self::PAD) as isize + k) as usize)]
                        .accum_backward(&view);
                }
            }
        }
        let mut output = np::Array3::<f32>::ones((b, w, d));
        for i in 0..self.width {
            let val = &self.grid[(self.depth - 1, i + Self::PAD)].backward();
            output
                .slice_mut(np::s![.., i, ..])
                .zip_mut_with(&val, |x, y| {
                    *x = *y;
                });
        }

        output
    }

    fn get_grads<'a>(&'a mut self) -> Vec<GradientParams<'a>> {
        let mut grads = Vec::new();
        grads.reserve_exact(self.grid.len());
        self.grid.iter_mut().for_each(|g| {
            g.get_grads(&mut grads);
        });
        grads
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
    let input = np::Array2::random((14, 16), Normal::new(0.0, 1.0).unwrap());
    let grad = np::Array2::ones((14, 16));

    node.accum(&input.view());
    println!("msg accum magnitude {}", maximum(&node.accum_msg.view()));
    let _y = node.forward();
    println!("msg accum magnitude {}", maximum(&node.accum_msg.view()));
    println!("msg magnitude {}", maximum(&_y.view()));
    node.accum_backward(&grad.view());
    let _grad = node.backward();
}

fn maximum<T: Float, D: Dimension>(a: &np::ArrayView<T, D>) -> T {
    a.fold(-T::max_value(), |a, b| a.max((*b).abs()))
}

#[test]
fn test_compute_grid() {
    let mut grid = Grid2D::new(32, 32, 4);
    let input = np::Array3::random((4, 32, 4), Normal::new(0.0, 1.0).unwrap());
    println!("inputs {}", maximum(&input.view()));
    let output = grid.forward(input);

    let (max_magnitude, has_nan) = output.fold((-f32::MAX, false), |a, b| {
        (a.0.max((*b).abs()), a.0.is_nan() || a.1)
    });
    println!("max magnitude {}, has nan {}", max_magnitude, has_nan);
    assert!(max_magnitude < 2.0);
    assert!(!has_nan);

    let grad = np::Array3::random((4, 32, 4), Normal::new(0.0, 1.0).unwrap());
    let dx = grid.backward(grad);

    let (max_magnitude, has_nan) = dx.fold((-f32::MAX, false), |a, b| {
        (a.0.max((*b).abs()), a.0.is_nan() || a.1)
    });
    println!("max magnitude {}, has nan {}", max_magnitude, has_nan);
}
