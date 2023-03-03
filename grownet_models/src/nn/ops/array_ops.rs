use arrayfire::*;
use arrayfire as af;

use super::Float;

pub fn reshape<T: Float>(x: &Array<T>, dims: Dim4) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let reshaped = moddims(x, dims);

    let orig_dims = x.dims();
    let df = move |grad: &Array<T>| {
        moddims(grad, orig_dims)
    };

    (reshaped, df)
}

pub fn reduce_sum<T: Float>(x: &Array<T>, dim: u64) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let xdim = x.dims();
    let y = sum(x, dim as i32);
    let df = move |grad: &Array<T>| {
        let outdim = match dim {
            0 => dim4!(xdim[0], 1, 1, 1),
            1 => dim4!(1, xdim[1], 1, 1),
            2 => dim4!(1, 1, xdim[2], 1),
            3 => dim4!(1, 1, 1, xdim[3]),
            _ => panic!()
        };
        tile(grad, outdim)
    };
    (y, df)
}

fn matmul<T: Float>(a: &Array<T>, b: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> (Array<T>, Array<T>)) {
    let y = af::matmul(a, b, MatProp::NONE, MatProp::NONE);
    let a1 = a.clone();
    let b1 = b.clone();
    let df = move |grad: &Array<T>| {
        let db = af::matmul(&a1, grad, MatProp::TRANS, MatProp::NONE);
        let da = af::matmul(grad, &b1, MatProp::NONE, MatProp::TRANS);
        (da, db)
    };
    (y, df)
}

#[test]
fn gradcheck_reducesum() {
    set_backend(Backend::CPU);
    use super::utils::grad_check;
    let x = randn::<f64>(dim4!(16, 1, 2, 1));
    grad_check(x, None, None, None, |x| reduce_sum(x, 0));
}

#[test]
fn gradcheck_matmul() {
    set_backend(Backend::CPU);
    use super::utils::grad_check;
    let a = randn::<f64>(dim4!(16, 14));
    let b = randn::<f64>(dim4!(14, 15));
    let b1 = b.clone();
    grad_check(a.clone(), None, None, None, move |x: &Array<f64>| {
        let (y, f) = matmul(x, &b1);
        let df = move |g: &Array<f64>| f(g).0;
        (y, df)
    });

    let a1 = a.clone();
    grad_check(b, None, None, None, move |x: &Array<f64>| {
        let (y, f) = matmul(&a1, x);
        let df = move |g: &Array<f64>| f(g).1;
        (y, df)
    });
}