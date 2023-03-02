use std::rc::Rc;

use arrayfire::{self as af, dim4, Array};
use super::Float;

pub fn relu<T: Float>(a: &Array<T>) -> (Array<T>, impl FnMut(&Array<T>) -> Array<T>) {
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

#[cfg(test)]
mod test {
    use arrayfire::*;
    use super::*;
    use super::super::utils::grad_check;
    const CHECKDIM: u64 = 7;
    #[test]
    fn grad_check_softmax() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(CHECKDIM));
        grad_check(x, None, None, None, softmax);
    }

    #[test]
    fn grad_check_relu() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(CHECKDIM));
        grad_check(x, None, None, None, relu);
    }
}