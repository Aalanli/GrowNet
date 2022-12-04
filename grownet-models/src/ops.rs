use ndarray::prelude::*;
use num::Float;

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
