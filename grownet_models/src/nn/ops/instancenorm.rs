use arrayfire as af;
use af::*;

use super::{Float, Param, utils::{ones, zeros}};
use crate::Flatten;

#[derive(Flatten)]
pub struct InstanceNorm2D<T: Float> {
    gamma: Param<T>,
    beta: Param<T>,
}

impl<T: Float> InstanceNorm2D<T> {
    pub fn new(channels: u64) -> Self {
        Self { 
            gamma: Param::new(ones(dim4!(1, 1, channels))), 
            beta: Param::new(zeros(dim4!(1, 1, channels)))
        }
    }

    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl Fn(&mut Self, &Array<T>) -> Array<T>) {
        let (y, df) = instancenorm2d(x);
        let out = add(&self.beta.w, &mul(&self.gamma.w, &y, true), true);
        let y1 = y.clone();
        let new_df = move |s: &mut Self, grad: &Array<T>| {
            s.beta.g += sum_except_channels(grad);
            s.gamma.g += sum_except_channels(&mul(&y1, grad, true));
            df(&mul(grad, &s.gamma.w, true))
        };
        (out, new_df)
    }
}

fn sum_except_channels<T: Float>(x: &Array<T>) -> Array<T> {
    sum(&sum(&sum(x, 0), 1), 3)
}

#[test]
fn test_instancenorm2d() {
    let x = randn!(28, 28, 3, 1);
    println!("{}", instancenorm2d(&x).0.dims());
    let mut resnet = InstanceNorm2D::new(3);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);
}

/// instancernorm along h*w dim, so expected input shape is [w, h, c, b]
pub fn instancenorm2d<T: Float>(x: &Array<T>) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
    let dims = x.dims();
    let rdims = dim4!(dims[0] * dims[1], 1, dims[2], dims[3]);
    let flat = moddims(x, rdims);
    let (n, f) = af_instancenorm(&flat, T::from(1e-6).unwrap(), 0);
    let out = moddims(&n, dims);
    let new_f = move |grad: &Array<T>| {
        let reshape = moddims(grad, rdims);
        let dx_pre = f(&reshape);
        let dx_post = moddims(&dx_pre, dims);
        dx_post  
    };
    (out, new_f)
}


/// norm along ith dimension
pub fn af_instancenorm<T: Float>(x: &Array<T>, eps: T, dim: u64) -> (Array<T>, impl Fn(&Array<T>) -> Array<T>) {
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

        add(&dl_dci, &s_dci, true)
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
    grad_check(x, Some(1e-7), None, None, |x| af_instancenorm(x, 1e-8, 0));

    let x = randn::<f64>(dim4!(4, 4));
    grad_check(x, Some(1e-7), None, None, |x| af_instancenorm(x, 1e-8, 1));
}

#[test]
fn gradcheck_instancenorm2d() {
    set_backend(Backend::CPU);
    use super::utils::grad_check;
    let x = randn::<f64>(dim4!(16, 16, 3, 1));
    grad_check(x, Some(1e-7), None, None, |x| instancenorm2d(x));
}
