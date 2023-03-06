use arrayfire::*;
use arrayfire as af;

use ndarray as nd;

use super::{Float, utils::grad_check};


/// expects logits to be of shape [N, B], and gtruth to be the same shape
pub fn cross_entropy<T: Float>(logits: &Array<T>, gtruth: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let (y, df) = super::activations::log_softmax(logits);
    let result = mean(&sum(&mul(&y, gtruth, false), 0), 1);

    let gt = gtruth.clone();
    let df1 = move |grad: &Array<T>| {
        let partial = mul(&gt, grad, true) / T::from(y.dims()[1]).unwrap();
        df(&partial)
    };
    (result, df1)
}

/// expects a 1D array of shape [B], outputs an array of shape [classes, B]
pub fn one_hot<T: Float>(x: Array<u32>, classes: u32) -> Array<T> {
    let mut output = super::utils::zeros::<T>(dim4!(classes as u64, x.dims()[0]));
    let ones = super::utils::ones::<T>(dim4!(x.dims()[0]));
    let indices = &x + classes * range::<u32>(dim4!(x.dims()[0]), 0);
    eval!(output[indices] = ones);
    output
}

#[test]
fn test_onehot() {
    let a = Array::new(&[1, 2, 3u32, 2, 1], dim4!(5, 1));
    let out = one_hot::<f32>(a, 4);
    print(&out);
}

#[test]
fn test_crossentropy() {
    set_backend(Backend::CPU);
    let a = randn::<f64>(dim4!(8));
    let gt = randn::<f64>(dim4!(8));

    grad_check(a, None, None, None, |x| { cross_entropy(x, &gt) })
}