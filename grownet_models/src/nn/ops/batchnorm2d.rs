use std::rc::Rc;
use arrayfire::*;
use arrayfire as af;
use super::{Float, Param, utils::{ones, zeros}};

fn af_batchnorm2d_forward<T: Float>(
    input: &Array<T>, 
    momentum: T,
    eps: T,
    variance: &Array<T>,
    gamma: &Array<T>,
    beta: &Array<T>
) -> [Array<T>; 5] {    
    let input: &Array<T> = &*input;
    let mut flat = reorder_v2(&input, 0, 1, Some(vec![3, 2]));
    flat = moddims(&flat, Dim4::new(&[flat.elements() as u64 / flat.dims().get()[3], flat.dims().get()[3], 1, 1]));
    let mean = af::mean(&flat, 0);
    let var = var_v2(&flat, VarianceBias::POPULATION, 0);
    //self.mb_mean = reorder(&mean, Dim4::new(&[0, 2, 1, 3]));
    let mb_mean = reorder_v2(&mean, 0, 2, Some(vec![1, 3]));
    //self.mb_variance = reorder(&var, Dim4::new(&[0, 2, 1, 3]));
    let mb_variance = reorder_v2(&var, 0, 2, Some(vec![1, 3]));
    mb_mean.eval();
    mb_variance.eval();

    // Update the training set mean and variance using running averages
    let vt: T = T::from(1.0).unwrap().sub(momentum);
    let pt: T = momentum.into();
    let new_mean = mul(&pt, &mean, false) + &mb_mean * vt;
    let new_variance = mul(&pt, variance, false) + &mb_variance * vt;
    new_mean.eval();
    new_variance.eval();

    let eps: T = eps.into();
    let normalized_input = div(&sub(input, &mb_mean, true), &sqrt(&add(&mb_variance, &eps, true)), true);
    normalized_input.eval();

    let out = add(&mul(gamma, &normalized_input, true), beta, true);

    let normalized_input = normalized_input;
    let mb_variance = mb_variance;

    [out, new_mean, new_variance, normalized_input, mb_variance]
}

fn af_batchnorm2d_backward<T: Float>(
    dz: &Array<T>,
    mb_variance: &Array<T>,
    normalized_input: &Array<T>,
    eps: T,
    new_variance: &Array<T>,
    gamma: &Array<T>,
) -> [Array<T>; 3] {
    let dgamma = sum(&sum(&sum(&mul(dz, normalized_input, true), 3), 1), 0);
    let dbeta = sum(&sum(&sum(dz, 3), 1), 0);
    

    // Compute the derivative of the loss wrt the variance
    // c1 corresponds to: input - mb_mean
    let c1 = mul(normalized_input, &sqrt(&add(mb_variance, &eps, true)), true);
    // c2 corresponds to: sqrt(variance + eps)
    let c2 = sqrt(&add(mb_variance, &eps, true));
    let fac = mul(&div(gamma, &(T::from(-2.0).unwrap()), true), &pow(&c2, &(T::from(-3.0).unwrap()), true), true);
    let dmb_variance = mul(&sum(&mul(dz, &c1, true), 3), &fac, true);

    // Compute the derivative of the loss wrt the mean
    let term1 = mul(&sum(&dz, 3), &sub(&(T::zero()), &div(gamma, &c2, true), true), true);
    let term2 = mul(&dmb_variance, &mean(&mul(&(T::from(-2.0).unwrap()), &c1, true), 3), true);
    let dmb_mean = add(&term1, &term2, true);

    // Compute the derivative of the loss wrt the normalized input
    let dnormalized_input = mul(dz, gamma, true);

    // nan_check(&dnormalized_input, "dnormal");
    // nan_check(&new_variance, "new var");
    // nan_check(&&sqrt(new_variance), "sqrt var");
    // Compute and return the derivative of the loss wrt the input
    let term1 = mul(&dnormalized_input, &div(&(T::one()), &sqrt(new_variance), true), true);
    // nan_check(&term1, "term1");
    let m = normalized_input.dims()[3] as f32;
    let m = T::from(m).unwrap();
    let term2 = mul(&dmb_variance, &mul(&(T::from(2.0).unwrap() / m), &c1, true), true);
    let term3 = div(&dmb_mean, &m, true);
    let dout = add(&term1, &add(&term2, &term3, true), true);

    [dout, dgamma, dbeta]
}

fn nan_check<T: Float>(a: &Array<T>, msg: &str) {
    if any_true_all(&isnan(a)).0 {
        panic!("{}", msg);
    }
}

pub struct BatchNorm2D<T: Float> {
    mean: Array<T>,
    variance: Array<T>,
    gamma: Param<T>,
    beta: Param<T>,
    momentum: T,
    eps: T,
}

impl<T: Float> BatchNorm2D<T> {
    pub fn new(channels: u64) -> Self {
        BatchNorm2D { 
            mean: zeros(dim4!(1, 1, channels, 1)), 
            variance: ones(dim4!(1, 1, channels, 1)), 
            gamma: Param::new(ones(dim4!(1, 1, channels, 1))), 
            beta: Param::new(zeros(dim4!(1, 1, channels, 1))),
            momentum: T::from(0.99).unwrap(),
            eps: T::from(1e-5).unwrap()
        }
    }

    pub fn forward(&mut self, input: &Array<T>) -> (Array<T>, impl FnMut(&mut Self, &Array<T>) -> Array<T>) {
        let input: &Array<T> = &*input;
        let mut flat = reorder_v2(&input, 0, 1, Some(vec![3, 2]));
        flat = moddims(&flat, Dim4::new(&[flat.elements() as u64 / flat.dims().get()[3], flat.dims().get()[3], 1, 1]));
        let mean = mean(&flat, 0);
        let var = var_v2(&flat, VarianceBias::POPULATION, 0);
        //self.mb_mean = reorder(&mean, Dim4::new(&[0, 2, 1, 3]));
        let mb_mean = reorder_v2(&mean, 0, 2, Some(vec![1, 3]));
        //self.mb_variance = reorder(&var, Dim4::new(&[0, 2, 1, 3]));
        let mb_variance = reorder_v2(&var, 0, 2, Some(vec![1, 3]));
        mb_mean.eval();
        mb_variance.eval();

        // Update the training set mean and variance using running averages
        let vt: T = T::from(1.0).unwrap().sub(self.momentum);
        let pt: T = self.momentum.into();
        self.mean = mul(&pt, &self.mean, false) + &mb_mean * vt;
        self.variance = mul(&pt, &self.variance, false) + &mb_variance * vt;
        self.mean.eval();
        self.variance.eval();

        let eps: T = self.eps.into();
        let normalized_input = div(&sub(input, &mb_mean, true), &sqrt(&add(&mb_variance, &eps, true)), true);
        normalized_input.eval();

        let out = add(&mul(&self.gamma.w, &normalized_input, true), &self.beta.w, true);

        let normalized_input = Rc::new(normalized_input);
        let mb_variance = Rc::new(mb_variance);

        let back_fn = move |s: &mut Self, dz: &Array<T>| {
            let dgamma = sum(&sum(&sum(&mul(dz, &*normalized_input, true), 3), 1), 0);
            let dbeta = sum(&sum(&sum(dz, 3), 1), 0);
    
            s.gamma.g += dgamma;
            s.beta.g += dbeta;

            // Compute the derivative of the loss wrt the variance
            // c1 corresponds to: input - mb_mean
            let c1 = mul(&*normalized_input, &sqrt(&add(&*mb_variance, &s.eps, true)), true);
            // c2 corresponds to: sqrt(variance + eps)
            let c2 = sqrt(&add(&*mb_variance, &s.eps, true));
            let fac = mul(&div(&s.gamma.w, &T::from(-2.0).unwrap(), true), &pow(&c2, &T::from(-3.0).unwrap(), true), true);
            let dmb_variance = mul(&sum(&mul(dz, &c1, true), 3), &fac, true);
    
            // Compute the derivative of the loss wrt the mean
            let term1 = mul(&sum(&dz, 3), &sub(&T::zero(), &div(&s.gamma.w, &c2, true), true), true);
            let term2 = mul(&dmb_variance, &af::mean(&mul(&T::from(-2.0).unwrap(), &c1, true), 3), true);
            let dmb_mean = add(&term1, &term2, true);
    
            // Compute the derivative of the loss wrt the normalized input
            let dnormalized_input = mul(dz, &s.gamma.w, true);
    
            // Compute and return the derivative of the loss wrt the input
            let term1 = mul(&dnormalized_input, &div(&T::one(), &sqrt(&s.variance), true), true);
            let m = T::from(normalized_input.dims()[3] as f32).unwrap();
            let term2 = mul(&dmb_variance, &mul(&T::from(2.0).unwrap().div(m), &c1, true), true);
            let term3 = div(&dmb_mean, &m, true);
            add(&term1, &add(&term2, &term3, true), true)
        };

        (out, back_fn)
    }
}

#[test]
fn test_forward_backward() {
    use super::utils::{grad_check, assign};
    set_backend(Backend::CPU);
    let dim = 2;
    let mut batchnorm = BatchNorm2D::new(4);
    let input = randn::<f64>(dim4!(dim, dim, 4, 1));
    let (x, mut f) = batchnorm.forward(&input);
    let mut testg = constant(0.0, x.dims());
    assign(&mut testg, 0, 1.0);
    af::print(&(f(&mut batchnorm, &testg)));
}

#[test]
fn gradcheck_batchnorm2d() {
    use super::utils::{grad_check, assign};
    set_backend(Backend::CPU);
    let dim = 4;
    let input = randn::<f64>(dim4!(dim, dim));
    let variance = abs(&randn::<f64>(dim4!(dim, dim)));
    let gamma = randn::<f64>(dim4!(dim, dim));
    let beta = randn::<f64>(dim4!(dim, dim));
    let momentum = 0.99;
    let eps = 1e-5;

    let test_dinput = |x: &Array<f64>| {
        let [out, _new_mean, new_variance, normalized_input, mb_variance] = 
            af_batchnorm2d_forward(x, momentum, eps, &variance, &gamma.clone(), &beta);
        let gammap = gamma.clone();
        let back_fn = move |grad: &Array<f64>| {
            let [dx, _dg, _db] = 
                af_batchnorm2d_backward(grad, &mb_variance, &normalized_input, eps, &new_variance, &gammap);
            dx
        };

        (out, back_fn)
    };

    let (tout, f) = test_dinput(&input); 
    print(&tout);
    let mut testg = constant(0.0, tout.dims());
    assign(&mut testg, 0, 1.0);
    print(&f(&testg));

    grad_check(input, None, None, None, test_dinput);    

}
