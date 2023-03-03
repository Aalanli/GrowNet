use arrayfire as af;
use af::*;

use super::Float;

/// norm along ith dimension
pub fn af_instancenorm2d<T: Float>(x: &Array<T>, eps: T, dim: u64) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let mu = mean(x, dim as i64);
    let ci = sub(x, &mu, true);
    let n = ci.dims()[dim as usize];
    let var = mean(&(&ci * &ci), dim as i64);
    let inv_std = div(&T::one(), &sqrt(&(var + eps)), true);
    let out = mul(&ci, &inv_std, true);

    let ci = ci.clone();
    let back_fn = move |grad: &Array<T>| {
        let d_std_inv_d_ci = { // Rn
            let temp = pow(&inv_std, &T::from(3.0).unwrap(), true) / T::from(n as f32).unwrap().neg();
            mul(&temp, &ci, true)
        };

        let dyi_dinv_std = sum(&mul(grad, &ci, true), dim as i32); // R
        let temp_dl_dci = mul(&d_std_inv_d_ci, &dyi_dinv_std, true);

        let dl_dci = add(&mul(grad, &inv_std, true), &temp_dl_dci, true);

        let s_dci = sum(&dl_dci, dim as i32) / T::from(n).unwrap().neg();

        dl_dci + s_dci
    };

    (out, back_fn)
}


fn std<T: Float>(xi: &Array<T>, dim: u64) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let xi1 = xi.clone();
    let yi = mean(&pow(xi, &T::from(2.0).unwrap(), true), dim as i64);
    let yi1 = sqrt(&yi);

    let yi2 = yi1.clone();
    let back_fn = move |grad: &Array<T>| {
        let n = T::from(xi1.dims()[dim as usize] as f32).unwrap();
        let a = div(&(&xi1 / n), &yi2, true);
        mul(&a, grad, true)
    };

    (yi1, back_fn)
}


#[test]
fn gradcheck_std() {
    set_backend(Backend::CPU);
    use super::utils::grad_check;
    let x = randn::<f64>(dim4!(16, 1));
    grad_check(x, Some(1e-7), None, None, |x| std(x, 0));
}


#[test]
fn gradcheck_instancenorm() {
    set_backend(Backend::CPU);
    use super::utils::grad_check;
    let x = randn::<f64>(dim4!(16, 1));
    grad_check(x, Some(1e-7), None, None, |x| af_instancenorm2d(x, 1e-8, 0));
}
