use std::rc::Rc;

use arrayfire::{self as af, dim4, Array};
use super::Float;

pub fn relu<T: Float>(a: &Rc<Array<T>>) -> (Rc<Array<T>>, impl FnMut(&Array<T>) -> Array<T>) {
    let aref: &Array<T> = &*a;
    let y = af::maxof(&af::constant(T::zero(), dim4!(1)), aref, true);

    let a = a.clone();
    let back_fn = move |grad: &Array<T>| {
        let aref: &Array<T> = &*a;
        let gate = af::ge(aref, &T::zero(), true);
        let dx = gate.cast() * grad;

        dx
    };

    (Rc::new(y), back_fn)
}

pub fn softmax<T: Float>(a: &Rc<Array<T>>) -> (Rc<Array<T>>, impl Fn(&Array<T>) -> Array<T>) {
    let a: &Array<T> = &*a;
    let shifted = af::sub(a, &af::max(a, 0), true);
    let exp = af::exp(&shifted);
    let result = af::div(&exp, &af::sum(&exp, 0), true);
    let out = Rc::new(result);

    let store = out.clone();
    let back_fn = move |grad: &Array<T>| {
        // grad and input should be the same size, so batch is false
        let dot_last = af::sum(&af::mul(grad, &*store, false), 0);
        let dx = af::mul(&*store, &af::sub(grad, &dot_last, true), false);
        dx
    };
    
    (out, back_fn)
}

use af::*;
/// compute the jacobian matrix using finite differences, only works if f is 'relatively' pure
pub fn compute_jacobian(x: Array<f64>, eps: f64, mut f: impl FnMut(&Array<f64>) -> Array<f64>) -> Array<f64> {
    af::set_backend(Backend::CPU);
    let x = x.clone();
    let nd = x.elements() as usize;
    let md = {
        let test_out = f(&x);
        test_out.elements() as usize
    };
    let jac = constant(0.0f64, dim4!(md as u64, nd as u64));

    x.eval();
    jac.eval();
    let eps2 = 2.0 * eps;
    unsafe {
        let ptr = x.device_ptr() as *mut f64;
        let jac_ptr = jac.device_ptr() as *mut f64;
        for i in 0..nd {
            let orig = *ptr.add(i);
            // perturbe once
            *ptr.add(i) += eps;
            
            let out1 = f(&x);

            // perturbe another time
            *ptr.add(i) = orig - eps;

            let out2 = f(&x);
            out1.eval();
            out2.eval();
            let optr1 = out1.device_ptr() as *mut f64;
            let optr2 = out2.device_ptr() as *mut f64;
            // dy/dx = (f(x + h) - f(x - h)) / 2h
            let col_offset = i * md;
            for j in 0..md {
                let dy_dxi = (*optr1.add(j) - *optr2.add(j)) / eps2;
                *jac_ptr.add(col_offset + j) = dy_dxi;
            }
            out1.unlock();
            out2.unlock();
            *ptr.add(i) = orig;
        }

        jac.unlock();
        x.unlock();
    }
    jac
}

/// checks the gradient by finite differences, first computing the jacobian
/// works only if f is pure in the sense the two calls with identical data
/// evalulates to the same thing
pub fn grad_check<FB>(x: Array<f64>, eps: f64, atol: Option<f64>, rtol: Option<f64>, mut f: impl FnMut(&Array<f64>) -> (Array<f64>, FB))
where FB: FnMut(&Array<f64>) -> Array<f64>
{
    af::set_backend(Backend::CPU);
    let x = x.clone();
    let jac_fn = |h: &Array<f64>| {
        f(h).0
    };

    // m * n
    let jacobian = compute_jacobian(x.clone(), eps, jac_fn);
    let (mut grad, mut grad_fn) = f(&x);
    grad = mul(&grad, &0.0, true);
    grad.eval();
    jacobian.eval();

    let md = grad.elements() as usize;

    unsafe {
        let gptr = grad.device_ptr() as *mut f64;
        for i in 0..md {
            *gptr.add(i) = 1.0;
            // compute gradient of the ith row
            let analytical = flat(&grad_fn(&grad));
            let numerical = flat(&slice_dim(&jacobian, 0, i as u64));
            if !is_close(&numerical, &analytical, atol, rtol) {
                af::print(&jacobian);
                panic!("jacobian mismatch on row {}", i);
            }
        }
    }

}

pub fn slice_dim<T: af::HasAfEnum>(a: &Array<T>, dim: u64, idx: u64) -> Array<T> {
    let seqs = match dim {
        0 => {[
                af::Seq::new(idx as f64, idx as f64, 1.0),
                af::Seq::default(),
                af::Seq::default(),
                af::Seq::default()
            ]}
        1 => {[
            af::Seq::default(),
            af::Seq::new(idx as f64, idx as f64, 1.0),
            af::Seq::default(),
            af::Seq::default()
        ]}
        2 => {[
            af::Seq::default(),
            af::Seq::default(),
            af::Seq::new(idx as f64, idx as f64, 1.0),
            af::Seq::default()
        ]}
        3 => {[
            af::Seq::default(),
            af::Seq::default(),
            af::Seq::default(),
            af::Seq::new(idx as f64, idx as f64, 1.0),
        ]},
        _ => panic!("dimension out of bounds")
    };

    af::index(a, &seqs)
}

pub fn is_close(a: &Array<f64>, b: &Array<f64>, atol: Option<f64>, rtol: Option<f64>) -> bool {
    let atol = atol.unwrap_or(1e-5);
    let rtol = rtol.unwrap_or(0.001);
    let diff = abs(&(b - a));
    let diff = ge(&diff, &(atol + rtol * abs(&a)), false);
    let any_true = any_true_all(&diff);
    !any_true.0
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_close() {
        let a: Array<f64> = randn(dim4!(3, 3, 3));
        let b = randn(dim4!(3, 3, 3));
        assert!(is_close(&a, &a, None, None));
        assert!(!is_close(&a, &b, None, None));
    }

    #[test]
    fn test_jacobian_linear() {
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(64));
        let f = |x: &Array<f64>| {
            x * 3.0
        };
        let jac = compute_jacobian(x, 1e-6, f);
        let analytical = diag_create(&constant(3.0, dim4!(64)), 0);
        assert!(is_close(&analytical, &jac, None, None));
    }

    #[test]
    fn test_jacobian_pointwise() {
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(64));
        let f = |x: &Array<f64>| {
            x * 3.0 + sin(&x) * exp(&x)
        };
        let jac = compute_jacobian(x.clone(), 1e-6, f);
        let ajac = 3.0 + (cos(&x) + sin(&x)) * exp(&x);
        let analytical = diag_create(&ajac, 0);
        assert!(is_close(&analytical, &jac, None, None));
    }

    #[test]
    fn test_gradcheck_linear() {
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(64));
        let f = |x: &Array<f64>| {
            let f = |g: &Array<f64>| {
                constant(3.0, g.dims()) * g
            };
            (x * 3.0, f)
        };

        grad_check(x, 1e-6, None, None, f);
    }

    fn pointwise_forward(x: &Array<f64>) -> (Array<f64>, impl FnMut(&Array<f64>) -> Array<f64>) {
        let xs = Rc::new(x.clone());
        let df = move |g: &Array<f64>| {
            g * (3.0 + (cos(&*xs) + sin(&xs)) * exp(&xs))
        };
        
        (x * 3.0 + sin(&x) * exp(&x), df)
    }
    #[test]
    fn test_gradcheck_pointwise() {
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(64));

        grad_check(x, 1e-6, None, None, pointwise_forward);
    }

    #[test]
    fn fail() {
        assert!(false);
    }


}
