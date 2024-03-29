use std::rc::Rc;

use arrayfire::{self as af, dim4, Array};
use super::Float;

pub fn relu<T: Float>(a: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let aref: &Array<T> = &*a;
    let y = af::maxof(&af::constant(T::zero(), dim4!(1)), aref, true);

    let a = a.clone();
    let back_fn = move |grad: &Array<T>| {
        let aref: &Array<T> = &a;
        let gate = af::ge(aref, &T::zero(), true);
        let dx = gate.cast() * grad;

        dx
    };

    (y, back_fn)
}

pub fn softmax<T: Float>(a: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let a: &Array<T> = &*a;
    let shifted = af::sub(a, &af::max(a, 0), true);
    let exp = af::exp(&shifted);
    let result = af::div(&exp, &af::sum(&exp, 0), true);
    let out = result;

    let store = out.clone();
    let back_fn = move |grad: &Array<T>| {
        // grad and input should be the same size, so batch is false
        let dot_last = af::sum(&af::mul(grad, &store, false), 0);
        let dx = af::mul(&store, &af::sub(grad, &dot_last, true), false);
        dx
    };
    
    (out, back_fn)
}

pub fn log_softmax<T: Float>(a: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let shifted = af::sub(a, &af::max(a, 0), true);
    let exp = af::exp(&shifted);
    let sum = af::sum(&exp, 0);

    let out = af::sub(&shifted, &af::log(&sum), true);
    let df = move |grad: &Array<T>| {
        let softmax = af::div(&exp, &sum, true);
        let dot_g = af::mul(&af::sum(grad, 0), &softmax, true);
        af::sub(grad, &dot_g, true)
    };

    (out, df)
}

#[cfg(test)]
mod test {
    use arrayfire::*;
    use super::*;
    use super::super::utils::af_grad_check;
    const CHECKDIM: u64 = 7;
    #[test]
    fn grad_check_softmax() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(CHECKDIM));
        af_grad_check(x, None, None, None, softmax);
    }

    #[test]
    fn grad_check_log_softmax() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(CHECKDIM));
        af_grad_check(x, None, None, None, log_softmax);
    }

    #[test]
    fn grad_check_relu() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(CHECKDIM));
        af_grad_check(x, None, None, None, relu);
    }
}