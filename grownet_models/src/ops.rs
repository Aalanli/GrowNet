use tch::Tensor;
use anyhow::{Result, Error};
use ndarray::prelude::*;
use num::Float;
use crate::datasets::data::*;

const EPSILON: f32 = 1e-5;

pub fn mean(x: &[f32]) -> f32 {
    x.iter().fold(0.0, |x, y| x + *y) / x.len() as f32
}

pub fn st_dev(x: &[f32], mu: Option<f32>) -> f32 {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let mut std = x.iter().fold(0.0, |x, y| x + (*y - mu).powf(2.0));
    std /= (x.len() - 1) as f32;
    std = std.sqrt();
    std
}

pub fn normalize(x: &[f32], y: &mut [f32], mu: Option<f32>, std: Option<f32>) -> (f32, f32) {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let std = std.map_or_else(|| st_dev(x, Some(mu)), |x| x);

    y.iter_mut().zip(x.iter()).for_each(|(y, x)| {
        *y = (*x - mu) / (std + EPSILON);
    });

    (mu, std)
}

pub fn dnormalize(grad: &[f32], x: &[f32], dy_dx: &mut [f32], mu: Option<f32>, std: Option<f32>) {
    let mu = mu.map_or_else(|| mean(x), |x| x);
    let std = std.map_or_else(|| st_dev(x, Some(mu)), |x| x);

    let nvar = 1.0 / (std + EPSILON);
    let mut dotx = 0.0;
    let mut sum_grad = 0.0;
    grad.iter().zip(x.iter()).for_each(|(x, y)| {
        sum_grad += *x;
        dotx += *x * *y;
    });

    let m = nvar * nvar / (std * (x.len() - 1) as f32);
    let m = (sum_grad * mu - dotx) / m;
    let b = sum_grad / (x.len() as f32) * nvar;
    grad.iter().zip(x.iter()).zip(dy_dx.iter_mut()).for_each(|((g, x), y)| {
        let x = *x;
        let g = *g;
        *y = (x - mu) * m - b + g * nvar;
    });
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn drelu(x: f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
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
