use af::{dim4, Array};
use arrayfire::{self as af};

fn main() {
    af::set_backend(af::Backend::CPU);
    let mut a = af::diag_create::<f32>(&af::randn(dim4!(5)), 0);
    
    af::print(&a);
    println!("{:?}", af::any_true_all(&a));
}

#[cfg(test)]
mod test {
    use model_lib::nn::ops::activations::*;
    use arrayfire::*;
    use arrayfire as af;
    use std::rc::Rc;
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

}
