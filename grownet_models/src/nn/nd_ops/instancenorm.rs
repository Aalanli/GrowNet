use ndarray as nd;
use nd::{prelude::*, RemoveAxis, Zip};

use num::Float;
use num_traits::FromPrimitive;

use super::*;

pub struct InstanceNorm<T, D> {
    ci: Array<T, D>,
    inv_sd: Array<T, D>,
    axis: usize,
}


pub fn var_axis<A, D>(x: &ArrayView<A, D>, axis: usize) -> (Array<A, D>, Array<A, D>)
where
    A: Float + FromPrimitive,
    D: RemoveAxis + Dimension,
{
    let n = A::from_usize(x.len_of(Axis(axis))).expect("Converting length to `A` must not fail.");
    let dof = n;
    let mut dim = x.raw_dim();
    dim.slice_mut()[axis] = 1;


    let mut mean = Array::<A, D>::zeros(dim.clone());
    let mut sum_sq = Array::<A, D>::zeros(dim);
    for (i, subview) in x.axis_iter(Axis(axis)).enumerate() {
        let count = A::from_usize(i + 1).expect("Converting index to `A` must not fail.");
        mean.iter_mut().zip(sum_sq.iter_mut()).zip(subview.iter())
            .for_each(|((mean, sum_sq), x)| {
                let delta = *x - *mean;
                *mean = *mean + delta / count;
                *sum_sq = (*x - *mean).mul_add(delta, *sum_sq);
            });
    }
    (sum_sq.mapv_into(|s| s / dof), mean)
}

pub fn norm_axis<A, D>(x: &ArrayView<A, D>, axis: usize) -> (Array<A, D>, InstanceNorm<A, D>)
where
    A: Float + FromPrimitive,
    D: Dimension + RemoveAxis,
{
    let eps = A::from(1e-6).unwrap();
    let (var, mu) = var_axis(x, axis);
    
    let inv_sd = var.mapv_into(|x| A::one() / (x + eps).sqrt());
    let ci = x - mu;
    let out = &ci * &inv_sd;
    (out, InstanceNorm { ci, inv_sd, axis })
}

pub fn dnorm_axis<A, D>(ctx: &InstanceNorm<A, D>, grad: &ArrayView<A, D>) -> Array<A, D>
where
    A: Float,
    D: Dimension + RemoveAxis,
{
    let dot_gi = dot_axis(&ctx.ci.view(), grad, ctx.axis);

    let mut dy_dc = Array::zeros(grad.raw_dim());
    let axis = Axis(ctx.axis);
    let n = A::from(grad.len_of(axis)).unwrap();

    Zip::from(&ctx.ci).and_broadcast(&ctx.inv_sd).and(grad).and_broadcast(&dot_gi).and(&mut dy_dc)
        .for_each(|ci, inv_sd, grad, dot_gi, dy_dc| {
            *dy_dc = ci.neg() / n * inv_sd.powi(3) * *dot_gi + *grad * *inv_sd;
        });

    let dy_dxi_t = mean_axis(&dy_dc.view(), axis.0);

    dy_dc - dy_dxi_t
}


#[test]
fn test_dnorm() {
    let x = randn64(8);
    let (_, ctx) = norm_axis(&x.view(), 0);
    let f = |x: &Array1<f64>| {
        norm_axis(&x.view(), 0).0
    };
    let df = |grad: &Array1<f64>| {
        dnorm_axis(&ctx, &grad.view())
    };

    grad_check(x, f, df, None, None, None).unwrap();

    let x = randn64((8, 2));
    let (_, ctx) = norm_axis(&x.view(), 1);
    let f = |x: &Array1<f64>| {
        norm_axis(&x.clone().into_shape((8, 2)).unwrap().view(), 1).0.into_shape(16).unwrap()
    };
    let df = |grad: &Array1<f64>| {
        dnorm_axis(&ctx, &grad.clone().into_shape((8, 2)).unwrap().view()).into_shape(16).unwrap()
    };

    grad_check(x.into_shape(16).unwrap(), f, df, None, None, None).unwrap();
}


