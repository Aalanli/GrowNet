use std::borrow::Borrow;

use tch::Tensor;
use anyhow::{Result, Error};
use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, rand, RandomExt};
use num::{Float, complex::ComplexFloat};
use crate::datasets::data::*;

const EPSILON: f32 = 1e-5;


/// computes the jacobian with finite-difference approximation
/// where f: R^n -> R^m, the jacobian is R^nxm, out_len is m
fn compute_jacobian(
    mut input: Array1<f64>,
    f: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: f64,
    out_len: Option<usize>) -> Array2<f64> {
    
    // dy/dx = lim h->0 (f(x + h) - f(x - h)) / (2h)
    let d_eps = 2.0 * epsilon;
    
    let n = input.len();
    let m = if let Some(x) = out_len {x} else {
        let test_out = f(&input);
        test_out.len()
    };
    
    // first construct the jacobian of f with finite difference method
    let mut jac = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let old = input[i];
        input[i] = old + epsilon;
        let diff1 = f(&input);
        input[i] = old - epsilon;
        let diff2 = f(&input);

        jac.index_axis_mut(Axis(0), i)
            .iter_mut()
            .zip(diff1.iter()).zip(diff2.iter())
            .for_each(|((x, d1), d2)| {
                *x = (*d1 - *d2) / d_eps;
            });
        input[i] = old;
    }

    jac
}

/// Expect that both f and df are pure functions
/// f: R^n -> R^m
/// df: R^m -> R^n, where the first argument is the gradient w.r.t. the image of f
/// grads are considered equal if the analytical gradient x, and pertubed gradient y
/// satisfies |x - y| <= atol + rtol * |y|,
/// epsilon is defaulted to 1e-6, atol is defaulted to 1e-5, rtol is defauled to 0.001
pub fn grad_check(
    input: Array1<f64>,
    f: impl Fn(&Array1<f64>) -> Array1<f64>, 
    df: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: Option<f64>,
    atol: Option<f64>,
    rtol: Option<f64>) -> Result<()> {
    
    // defaults
    let epsilon = epsilon.unwrap_or(1e-6);
    let atol = atol.unwrap_or(1e-5);
    let rtol = rtol.unwrap_or(0.001);

    let test_out = f(&input);
    let n = input.len();
    let m = test_out.len();
    let mut dy_dx = Array1::<f64>::zeros(m);
    let test_grad = df(&dy_dx);
    if test_grad.len() != n {
        return Err(Error::msg(format!("f maps R{n} to R{m}, but df maps R{m} to R{}", test_grad.len())));
    }
    let jacobian = compute_jacobian(input, f, epsilon, Some(m));
    
    for i in 0..m {
        dy_dx[i] = 1.0;
        let grad = df(&dy_dx);
        let diff = jacobian.index_axis(Axis(1), i);
        for (x, y) in grad.iter().zip(diff.iter()) {
            if (x - y).abs() > atol + rtol * y.abs() {
                return Err(Error::msg(format!("jacobian mismatch on column {i} \n jacobian computed: \n {diff} \n analytical computed: \n {grad}")));
            }
        }
        dy_dx[i] = 0.0;
    }

    Ok(())
}


#[test]
fn test_grad_check() {
    fn test_grad_check_pointwise(
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        elems: usize,
    ) {
        let f = |x: &Array1<f64>| {
            x.map(|x| f(*x))
        };
        
        let x = Array1::random(elems, Normal::new(0.0, 1.0).unwrap());
        let df_ = |grad: &Array1<f64>| {
            x.iter().zip(grad.iter()).map(|(x, g)| {
                g * df(*x)
            }).collect()
        };
        grad_check(x.clone(), f, df_, None, None, None).unwrap();
    }

    // println!("polynomial");
    test_grad_check_pointwise(|x| 2.0 * x + 3.0 * x * x + 1.0, |x| 2.0 + 6.0 * x, 64);
    // println!("trig, exp");
    test_grad_check_pointwise(|x| x.sin() + x.exp(), |x| x.cos() + x.exp(), 64);
    // println!("product rule");
    test_grad_check_pointwise(|x| x.sin() * x.exp(), |x| x.cos() * x.exp() + x.sin() * x.exp(), 64);
    // println!("chain rule");
    test_grad_check_pointwise(|x| (x.cosh()).ln(), |x| x.tanh(), 64);
}

pub fn mean<T: Float>(x: &[T]) -> T {
    x.iter().fold(T::zero(), |x, y| x + *y) / T::from(x.len()).unwrap()
}

pub fn st_dev<T: Float>(x: &[T], mu: Option<T>) -> T {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let mut std = x.iter().fold(T::zero(), |x, y| x + (*y - mu).powi(2));
    std = std / T::from(x.len() - 1).unwrap();
    std = std.sqrt();
    std
}

pub fn normalize<T: Float>(x: &[T], y: &mut [T], mu: Option<T>, std: Option<T>) -> (T, T) {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let std = std.map_or_else(|| st_dev(x, Some(mu)), |x| x);

    y.iter_mut().zip(x.iter()).for_each(|(y, x)| {
        *y = (*x - mu) / (std + T::from(EPSILON).unwrap());
    });

    (mu, std)
}

pub fn dnormalize<T: Float>(grad: &[T], x: &[T], dy_dx: &mut [T], mu: Option<T>, std: Option<T>) {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let std = std.map_or_else(|| st_dev(x, Some(mu)), |x| x);

    let nvar = T::one() / (std + T::from(EPSILON).unwrap());
    let mut dotx = T::zero();
    let mut sum_grad = T::zero();
    grad.iter().zip(x.iter()).for_each(|(x, y)| {
        sum_grad = sum_grad + *x;
        dotx = dotx + *x * *y;
    });

    let m = nvar * nvar / (std * T::from(x.len() - 1).unwrap());
    let m = (sum_grad * mu - dotx) / m;
    let b = sum_grad / T::from(x.len()).unwrap() * nvar;
    grad.iter().zip(x.iter()).zip(dy_dx.iter_mut()).for_each(|((g, x), y)| {
        let x = *x;
        let g = *g;
        *y = (x - mu) * m - b + g * nvar;
    });
}

#[test]
fn grad_check_normalize() {
    let f = |x: &Array1<f64>| {
        let mut y = x.clone();
        normalize(x.as_slice().unwrap(), y.as_slice_mut().unwrap(), None, None);
        y
    };

    let df = |grad: &Array1<f64>, x: &Array1<f64>| {
        let mut dy_dx = grad.clone();
        dnormalize(grad.as_slice().unwrap(), x.as_slice().unwrap(), dy_dx.as_slice_mut().unwrap(), None, None);
        dy_dx
    };

    let x = Array1::random(16, Normal::new(0.0, 1.0).unwrap());
    let df_ = |grad: &Array1<f64>| {
        df(grad, &x)
    };
    grad_check(x.clone(), f, df_, None, None, None).unwrap();
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn drelu(x: f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
}

pub fn softmax<'a, T: Float>(x: ArrayView1<'a, T>, mut y: ArrayViewMut1<'a, T>) {
    let max = x.fold(-T::max_value(), |x, v| x.max(*v));
    let mut sum = T::zero();
    y.zip_mut_with(&x, |y, x| {
        *y = (*x - max).exp();
        sum = sum + *y;
    });
    y.iter_mut().for_each(|x| *x = *x / sum);
}

pub fn d_softmax<'a, T: Float>(
    grad: ArrayView1<'a, T>, 
    y: ArrayView1<'a, T>, 
    mut dy_dx: ArrayViewMut1<T>,
    ) {
    
    let dot = grad.iter().zip(y.iter()).fold(T::zero(), |v, (a, b)| {
        v + *a * *b
    });

    dy_dx.iter_mut().zip((grad.iter()).zip(y.iter())).for_each(|(v, (g, y))| {
        *v = *y * (*g - dot);
    });
}

#[test]
fn gradcheck_softmax() {
    let f = |x: &Array1<f64>| {
        let mut y = x.clone();
        softmax(x.view(), y.view_mut());
        y
    };
    let x = Array1::random(16, 
        Normal::new(0.0, 1.0).unwrap());
    
    let y = f(&x);
    let df = |grad: &Array1<f64>| {
        let mut dy_dx = x.clone();
        d_softmax(grad.view(), y.view(), dy_dx.view_mut());
        dy_dx
    };

    grad_check(x.clone(), f, df, None, None, None).unwrap();
}


fn maximum<T: Float, D: Dimension>(a: &ArrayView<T, D>) -> T {
    a.fold(-T::max_value(), |a, b| a.max((*b).abs()))
}

/// Converts an image tensor NCWH to an image array of shape NWHC
pub fn convert_image_tensor(t: &Tensor) -> Result<Array<f32, Ix4>> {
    let dims: Vec<_> = t.size().iter().map(|x| *x as usize).collect();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::msg("tensor shape, expect NCWH"));
    }
    let t = t.to_device(tch::Device::Cpu).to_dtype(tch::Kind::Float, false, true);
    // convert to channels last
    let t = t.permute(&[0, 2, 3, 1]);
    let ptr = t.as_ptr() as *const f32;
    let arr = unsafe{ ArrayView4::<f32>::from_shape_ptr((dims[0], dims[1], dims[2], dims[3]), ptr).to_owned() };
    Ok(arr)
}

pub fn convert_image_array(a: &ArrayView4<f32>) -> Result<Tensor> {
    let slice = if let Some(a) = a.as_slice() {a} else {return Err(Error::msg("array should be contiguous"));};
    let dim = a.dim();
    let ts = Tensor::of_slice(slice).reshape(&[dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64]);
    Ok(ts)
}

/// user must ensure that internal type of tensor match T
pub unsafe fn ts_to_vec<T: Clone>(t: &Tensor) -> Vec<T> {
    let t = t.to_device(tch::Device::Cpu);
    let len = t.size().iter().fold(1, |x, y| x * y) as usize;
    let ptr = t.as_ptr() as *const T;
    
    let s = &*std::ptr::slice_from_raw_parts(ptr, len);
    s.to_vec()
}


/// Assumes that all the 3d arrays have the same size, this function
/// stacks all the images in the first dimension. [W, H, C] -> [B, W, H, C]
pub fn concat_im_size_eq(imgs: &[&Array3<f32>]) -> Image {
    let whc = imgs[0].dim();
    let b = imgs.len();
    let mut img = Array4::<f32>::zeros((b, whc.0, whc.1, whc.2));
    for i in 0..b {
        let mut smut = img.slice_mut(s![i, .., .., ..]);
        smut.zip_mut_with(imgs[i], |a, b| {
            *a = *b;
        });
    }
    Image { image: img }
}


#[test]
fn normalize_grad() {
    use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};

    let dim = 16;
    let x: Array1<f32> = Array::random((dim,), Normal::new(0.0, 1.0).unwrap());
    //let grad: Array1<f32> = Array1::ones([dim]);
    //let mut dy_dx: Array1::<f32> = Array1::ones([dim]);
    //let mut y: Array1::<f32> = Array1::ones([dim]);
    
    let mut normbuf = Array1::zeros(x.dim());
    normalize(x.as_slice().unwrap(), normbuf.as_slice_mut().unwrap(), None, None);

    println!("max norm {}", maximum(&normbuf.view()));
}
