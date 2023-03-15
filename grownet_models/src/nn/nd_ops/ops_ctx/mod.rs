use ndarray::linalg::general_mat_mul;

use super::*;
pub use super::context::*;

pub mod norm;
pub use norm::*;

pub fn randn<'a, T, D, Sh, Ctx>(ctx: &'a Ctx, dim: Sh) -> ArrayViewMut<'a, T, D> 
where T: Float, D: Dimension, Sh: IntoDimension<Dim=D> + Clone, Ctx: ArrayCtx<T>, StandardNormal: Distribution<T> {
    let mut buf = ctx.empty(dim);
    let mut rng = thread_rng();
    let sampler = Normal::new(T::zero(), T::one()).unwrap();
    buf.map_inplace(|x| {
        *x = sampler.sample(&mut rng);
    });
    buf
}


pub fn randu<'a, T, D, Sh, Ctx>(ctx: &'a Ctx, dim: Sh) -> ArrayViewMut<'a, T, D> 
where T: Float, D: Dimension, Sh: IntoDimension<Dim=D> + Clone, Ctx: ArrayCtx<T>, T: SampleUniform {
    let mut buf = ctx.empty(dim);
    let mut rng = thread_rng();
    let sampler = Uniform::new(T::zero(), T::one());
    buf.map_inplace(|x| {
        *x = sampler.sample(&mut rng);
    });
    buf
}


pub fn uniop<'a, T, D, Ctx>(ctx: &'a Ctx, a: &ArrayView<T, D>, f: impl Fn(T) -> T) -> ArrayViewMut<'a, T, D> 
where T: Float, D: Dimension, Ctx: ArrayCtx<T> 
{
    let mut buf = ctx.empty(a.raw_dim());

    nd::Zip::from(&mut buf).and(a)
        .for_each(|y, a| {
            *y = f(*a);
        });

    buf
}

pub fn binop<'a, T, D, Ctx>(ctx: &'a Ctx, a: &ArrayView<T, D>, b: &ArrayView<T, D>, f: impl Fn(T, T) -> T) -> ArrayViewMut<'a, T, D> 
where T: Float, D: Dimension, Ctx: ArrayCtx<T> 
{
    let mut buf = ctx.empty(a.raw_dim());

    nd::Zip::from(&mut buf).and(a).and_broadcast(b)
        .for_each(|y, a, b| {
            *y = f(*a, *b);
        });

    buf
}

pub fn add<'a, T: Float, D: Dimension, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView<T, D>, b: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
    binop(ctx, a, b, |a, b| a + b)
}

pub fn sub<'a, T: Float, D: Dimension, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView<T, D>, b: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
    binop(ctx, a, b, |a, b| a - b)
}

pub fn mul<'a, T: Float, D: Dimension, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView<T, D>, b: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
    binop(ctx, a, b, |a, b| a * b)
}

pub fn div<'a, T: Float, D: Dimension, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView<T, D>, b: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
    binop(ctx, a, b, |a, b| a / b)
}


pub fn permute<'a, T: Float, D: Dimension, Sh: IntoDimension<Dim = D> + Clone, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView<T, D>, dim: Sh) -> ArrayViewMut<'a, T, D> {
    let a = a.clone().permuted_axes(dim.clone());
    let mut buf = ctx.empty(a.raw_dim());
    buf.assign(&a);
    buf
}

pub fn matmul<'a, T: Float + 'static, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, a: &ArrayView2<T>, b: &ArrayView2<T>) -> ArrayViewMut2<'a, T> {
    let dim_a = a.raw_dim();
    let dim_b = b.raw_dim();
    let mut buf = ctx.empty([dim_a[0], dim_b[1]]);
    general_mat_mul(T::one(), a, b, T::zero(), &mut buf);
    buf
}

pub fn dmatmul<'a, T: Float + 'static, Ctx: ArrayCtx<T>>(ctx: &'a Ctx, grad: &ArrayView2<T>, a: &ArrayView2<T>, b: &ArrayView2<T>) -> (ArrayViewMut2<'a, T>, ArrayViewMut2<'a, T>) {
    let dim_a = a.raw_dim();
    let dim_b = b.raw_dim();
    let mut da = ctx.empty(dim_a);
    let mut db = ctx.empty(dim_b);

    let a = permute(ctx, a, [1, 0]);
    let b = permute(ctx, b, [1, 0]);

    general_mat_mul(T::one(), &a, grad, T::zero(), &mut da);
    general_mat_mul(T::one(), grad, &b, T::zero(), &mut db);
    (da, db)
}

pub fn dot_axis<'a, A: Float, D: Dimension + RemoveAxis, Ctx: ArrayCtx<A>>(ctx: &'a Ctx, x: &ArrayView<A, D>, y: &ArrayView<A, D>, axis: usize) -> ArrayViewMut<'a, A, D> {
    let mut buf = unit_axis(ctx, x.raw_dim(), axis);
    for (view_a, view_b) in x.axis_iter(Axis(axis)).zip(y.axis_iter(Axis(axis))) {
        for ((z, a), b) in buf.iter_mut().zip(view_a.iter()).zip(view_b.iter()) {
            *z = a.mul_add(*b, *z);
        }
    }
    buf
}

pub fn mean_axis<'a, A: Float, D: Dimension + RemoveAxis, Ctx: ArrayCtx<A>>(ctx: &'a Ctx, x: &ArrayView<A, D>, axis: usize) -> ArrayViewMut<'a, A, D> {
    let mut buf = unit_axis(ctx, x.raw_dim(), axis);
    let n = A::from(x.len_of(Axis(axis))).unwrap();
    for view_a in x.axis_iter(Axis(axis)) {
        for (z, a) in buf.iter_mut().zip(view_a.iter()) {
            *z = *z + *a;
        }
    }
    buf.mapv_into(|x| x / n)
}

pub fn unit_axis<'a, D: Dimension, F: Float, Ctx: ArrayCtx<F>>(ctx: &'a Ctx, mut dim: D, i: usize) -> ArrayViewMut<'a, F, D> {
    dim.slice_mut()[i] = 1;
    ctx.zeros(dim)
}

pub fn view_immut<A, D: Dimension>(view: ArrayViewMut<A, D>) -> ArrayView<A, D> {
    let dim = view.raw_dim();
    let ptr = view.as_ptr();
    unsafe { ArrayView::from_shape_ptr(dim, ptr) }
}

pub fn var_axis<'a, A, D, Ctx: ArrayCtx<A>>(ctx: &'a Ctx, x: &ArrayView<A, D>, axis: usize) -> (ArrayViewMut<'a, A, D>, ArrayViewMut<'a, A, D>)
where
    A: Float + FromPrimitive,
    D: RemoveAxis + Dimension,
{
    let n = A::from_usize(x.len_of(Axis(axis))).expect("Converting length to `A` must not fail.");
    let dof = n;
    let mut dim = x.raw_dim();
    dim.slice_mut()[axis] = 1;


    let mut mean = ctx.empty(dim.clone());
    let mut sum_sq = ctx.empty(dim);
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
