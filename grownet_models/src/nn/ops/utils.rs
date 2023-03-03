use arrayfire::*;
use arrayfire as af;
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::Float;

pub fn scaled_uniform<T: Float>(lower_bound: T, upper_bound: T, dims: Dim4) -> Array<T> {
    constant(lower_bound, dims) + constant(upper_bound.sub(lower_bound), dims) * randu::<T>(dims)
}

pub fn scaled_normal<T: Float>(mean: T, standard_deviation: T, dims: Dim4) -> Array<T> {
    constant(standard_deviation, dims) * randn::<T>(dims) + constant(mean, dims)
}

pub fn ones<T: Float>(dims: Dim4) -> Array<T> {
    constant(T::one(), dims)
}

pub fn zeros<T: Float>(dims: Dim4) -> Array<T> {
    constant(T::zero(), dims)
}


pub fn assign(a: &mut Array<f64>, i: usize, val: f64) {
    assert!(a.get_backend() == Backend::CPU);
    a.eval();
    unsafe {
        let ptr = a.device_ptr() as *mut f64;
        *ptr.add(i) = val;
        a.unlock();
    }
}

fn get(a: &mut Array<f64>, i: usize) -> f64 {
    assert!(a.get_backend() == Backend::CPU);
    a.eval();
    unsafe {
        let ptr = a.device_ptr() as *mut f64;
        let orig = *ptr.add(i);
        a.unlock();
        orig
    }
}

/// compute the jacobian matrix using finite differences, only works if f is 'relatively' pure
pub fn compute_jacobian(x: Array<f64>, eps: f64, mut f: impl FnMut(&Array<f64>) -> Array<f64>) -> Array<f64> {
    af::set_backend(Backend::CPU);
    let mut x = x.clone();
    let nd = x.elements() as usize;
    let md = {
        let test_out = f(&x);
        test_out.elements() as usize
    };
    let mut jac = constant(0.0f64, dim4!(md as u64, nd as u64));

    let eps2 = 2.0 * eps;

    for i in 0..nd {
        let orig = get(&mut x, i);
        // perturbe once
        assign(&mut x, i, orig + eps);
        let out1 = f(&x);

        // perturbe another time
        assign(&mut x, i, orig - eps);
        let out2 = f(&x);
        // print(&(&out1 - &out2));

        let dydxi = (out1 - out2) / eps2;
        set_col(&mut jac, &flat(&dydxi), i as i64);
        assign(&mut x, i, orig);
    }

    jac
}

/// checks the gradient by finite differences, first computing the jacobian
/// works only if f is pure in the sense the two calls with identical data
/// evalulates to the same thing
pub fn grad_check<FB>(x: Array<f64>, eps: Option<f64>, atol: Option<f64>, rtol: Option<f64>, mut f: impl FnMut(&Array<f64>) -> (Array<f64>, FB))
where FB: FnMut(&Array<f64>) -> Array<f64>
{
    af::set_backend(Backend::CPU);
    let x = x.clone();
    let jac_fn = |h: &Array<f64>| {
        f(h).0
    };

    let eps = eps.unwrap_or(1e-6);
    // m * n
    let jacobian = compute_jacobian(x.clone(), eps, jac_fn);
    let (mut grad, mut grad_fn) = f(&x);
    grad = mul(&grad, &0.0, true);
    grad.eval();
    jacobian.eval();

    let md = grad.elements() as usize;

    for i in 0..md {
        assign(&mut grad, i, 1.0);
        // compute gradient of the ith row
        let analytical = flat(&grad_fn(&grad));
        let numerical = flat(&row(&jacobian, i as i64));
        if !is_close(&numerical, &analytical, atol, rtol) {
            println!("jacobian mismatch on row {}", i);
            println!("numerical | analytical");
            let cat = join(1, &numerical, &analytical);
            af::print(&cat);
            if jacobian.elements() < 64 {
                println!("full jacobian");
                af::print(&jacobian);
            }
            panic!();
        }
        assign(&mut grad, i, 0.0);
    }
    

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
    use std::rc::Rc;
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
        let diml = 7;
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(diml));
        let f = |x: &Array<f64>| {
            x * 3.0
        };
        x.eval();
        let jac = compute_jacobian(x, 1e-4, f);
        let analytical = diag_create(&constant(3.0, dim4!(diml)), 0);
        //af::print(&jac);
        //af::print(&analytical);
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
        let x = randn::<f64>(dim4!(7));
        let f = |x: &Array<f64>| {
            let f = |g: &Array<f64>| {
                constant(3.0, g.dims()) * g
            };
            (x * 3.0, f)
        };

        grad_check(x, None, None, None, f);
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

        grad_check(x, None, None, None, pointwise_forward);
    }

    fn stateful_forward(x: &Array<f64>) -> (Array<f64>, impl FnMut(&Array<f64>) -> Array<f64>) {
        let xs = x.clone();
        let y = x * 3.0 + sin(&x) * exp(&x);

        let y1 = y.clone();
        let df = move |g: &Array<f64>| {
            let h = (&y + g) - &y;
            h * (3.0 + (cos(&xs) + sin(&xs)) * exp(&xs))
        };
        
        (y1, df)
    }
    #[test]
    fn test_stateful() {
        af::set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(64));

        grad_check(x, None, None, None, stateful_forward);
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

    #[test]
    fn test_softmax() {
        set_backend(Backend::CPU);
        let x = randn::<f64>(dim4!(16));
        grad_check(x, None, None, None, softmax);
    }


    
}
