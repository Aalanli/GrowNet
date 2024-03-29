use anyhow::{Error, Result};
use ndarray as nd;
use nd::{prelude::*, RemoveAxis, Zip, linalg::general_mat_mul};
use num::Float;
use ndarray_rand::{rand, rand_distr::Normal, RandomExt};


/// computes the jacobian with finite-difference approximation
/// where f: R^n -> R^m, the jacobian is R^nxm, out_len is m
fn compute_jacobian(
    mut input: Array1<f64>,
    f: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: f64,
    out_len: Option<usize>,
) -> Array2<f64> {
    // dy/dx = lim h->0 (f(x + h) - f(x - h)) / (2h)
    let d_eps = 2.0 * epsilon;

    let n = input.len();
    let m = if let Some(x) = out_len {
        x
    } else {
        let test_out = f(&input);
        test_out.len()
    };

    // first construct the jacobian of f with finite difference method
    let mut jac = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let old = input[i];
        input[i] = old + epsilon;
        let diff1 = f(&input);
        input[i] = old - epsilon;
        let diff2 = f(&input);

        jac.index_axis_mut(Axis(0), i)
            .iter_mut()
            .zip(diff1.iter())
            .zip(diff2.iter())
            .for_each(|((x, d1), d2)| {
                *x = (*d1 - *d2) / d_eps;
            });
        input[i] = old;
    }

    jac
}

/// Expect that both f and df are pure functions
/// f: R^n -> R^m
/// df: R^m -> R^n, where the first argument is the gradient w.r.t. the image of f
/// grads are considered equal if the analytical gradient x, and pertubed gradient y
/// satisfies |x - y| <= atol + rtol * |y|,
/// epsilon is defaulted to 1e-6, atol is defaulted to 1e-5, rtol is defauled to 0.001
pub fn grad_check(
    input: Array1<f64>,
    f: impl Fn(&Array1<f64>) -> Array1<f64>,
    df: impl Fn(&Array1<f64>) -> Array1<f64>,
    epsilon: Option<f64>,
    atol: Option<f64>,
    rtol: Option<f64>,
) -> Result<()> {
    // defaults
    let epsilon = epsilon.unwrap_or(1e-6);
    let atol = atol.unwrap_or(1e-5);
    let rtol = rtol.unwrap_or(0.001);

    let test_out = f(&input);
    let n = input.len();
    let m = test_out.len();
    let mut dy_dx = Array1::<f64>::zeros(m);
    let test_grad = df(&dy_dx);
    if test_grad.len() != n {
        return Err(Error::msg(format!(
            "f maps R{n} to R{m}, but df maps R{m} to R{}",
            test_grad.len()
        )));
    }
    let jacobian = compute_jacobian(input, f, epsilon, Some(m));

    for i in 0..m {
        dy_dx[i] = 1.0;
        let grad = df(&dy_dx);
        let diff = jacobian.index_axis(Axis(1), i);
        for (x, y) in grad.iter().zip(diff.iter()) {
            if (x - y).abs() > atol + rtol * y.abs() {
                return Err(Error::msg(format!("jacobian mismatch on column {i} \n jacobian computed: \n {diff} \n analytical computed: \n {grad}")));
            }
        }
        dy_dx[i] = 0.0;
    }

    Ok(())
}

pub fn isclose<D: Dimension, A: Float>(a: &Array<A, D>, b: &Array<A, D>) -> bool {
    let rtol = A::from(1e-5).unwrap();
    let atol = A::from(1e-8).unwrap();
    for (i, j) in a.iter().zip(b.iter()) {
        if !(*i - *j).le(&(atol + rtol * j.abs())) {
            return false;
        }
    }
    true
}
