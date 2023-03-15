use super::*;

pub struct InstanceNorm<T, D> {
    ci: ArrId<T, D>,
    inv_sd: ArrId<T, D>,
    axis: usize,
}

pub fn norm_axis<'a, A: Float + FromPrimitive, D: Dimension + RemoveAxis, Ctx: ArrayCtx<A>>(ctx: &'a Ctx, x: &ArrayView<A, D>, axis: usize) -> (ArrayViewMut<'a, A, D>, InstanceNorm<A, D>) {
    let eps = A::from(1e-6).unwrap();
    let (var, mu) = var_axis(ctx, x, axis);

    let inv_sd = var.mapv_into(|x| A::one() / (x + eps).sqrt());
    let mut ci = ctx.clone(x);
    
    let mut out = ctx.empty(ci.raw_dim());

    nd::Zip::from(&mut ci).and(&mut out).and_broadcast(&mu).and_broadcast(&inv_sd)
        .for_each(|ci, out, mu, inv_sd| {
            *ci = *ci - *mu;
            *out = *ci * *inv_sd;
        });
    
    let id = InstanceNorm {
        ci: ctx.id(ci),
        inv_sd: ctx.id(inv_sd),
        axis
    };
    (out, id)
}


pub fn dnorm_axis<'a, A: Float + FromPrimitive, D: Dimension + RemoveAxis, Ctx: ArrayCtx<A>>(ctx: &'a Ctx, ictx: &InstanceNorm<A, D>, grad: &ArrayView<A, D>) -> ArrayViewMut<'a, A, D> {
    let ci = ctx.from_id(&ictx.ci);
    let inv_sd = ctx.from_id(&ictx.inv_sd);

    let dot_gi = dot_axis(ctx, &ci, grad, ictx.axis);

    let mut dy_dc = ctx.empty(grad.raw_dim());
    let axis = Axis(ictx.axis);
    let n = A::from(grad.len_of(axis)).unwrap();

    nd::Zip::from(&ci).and_broadcast(&inv_sd).and(grad).and_broadcast(&dot_gi).and(&mut dy_dc)
        .for_each(|ci, inv_sd, grad, dot_gi, dy_dc| {
            *dy_dc = ci.neg() / n * inv_sd.powi(3) * *dot_gi + *grad * *inv_sd;
        });

    
    let dy_dxi_t = mean_axis(ctx,&dy_dc.view(), axis.0);

    nd::Zip::from(&mut dy_dc).and_broadcast(&dy_dxi_t).for_each(|y, x| { *y = *y - *x; } );

    dy_dc
}


#[test]
fn test_dnorm() {
    let ctx = FlatCtx::<f64>::new(512 * 1024);

    let x = randn(&ctx, 8);
    let (_, ictx) = norm_axis(&ctx, &x.view(), 0);
    let f = |x: &Array1<f64>| {
        norm_axis(&ctx,&x.view(), 0).0.view().into_owned()
    };
    let df = |grad: &Array1<f64>| {
        dnorm_axis(&ctx, &ictx, &grad.view()).into_owned()
    };

    grad_check(x.into_owned(), f, df, None, None, None).unwrap();

}

