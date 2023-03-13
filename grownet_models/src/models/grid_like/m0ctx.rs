use nd::IntoDimension;
use nd::linalg::general_mat_mul;
use ndarray::prelude::*;
use ndarray::{self as nd, RemoveAxis};
use ndarray_rand::{rand, rand::thread_rng, rand_distr::{Normal, Distribution, StandardNormal}, RandomExt};

use num::{Float, FromPrimitive};


pub trait ArrayCtx<T: Float> {
    fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D> + Clone>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D>;
    
    fn clone<'a, D: Dimension>(&'a self, xs: &ArrayView<'a, T, D>) -> ArrayViewMut<'a, T, D> {
        let mut empty = self.empty(xs.raw_dim());
        empty.assign(&xs);
        empty
    }

    fn zeros<'a, D: Dimension, Sh: IntoDimension<Dim = D> + Clone>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let mut arr = self.empty(dim);
        arr.fill(T::zero());
        arr
    }

    fn clear(&mut self);
}



pub struct Ctx<T> {
    buf: Vec<T>,
}

impl<T: Float> Ctx<T> {
    pub fn new(cap: usize) -> Self {
        let mut buf = Vec::new();
        buf.reserve_exact(cap);
        Ctx { buf }
    }

    unsafe fn reserve(&self, nelem: usize) -> usize {
        let cap = self.buf.capacity();
        let len = self.buf.len();
        if len + nelem > cap {
            panic!("not enough memory")
        }
        (&mut *(&self.buf as *const Vec<T> as *mut Vec<T>)).set_len(len + nelem);
        len
    }
    
    /// panics if there is not enough space
    pub fn empty<'a, D: Dimension, Sh: GDim<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_shape();
        let raw_dim = dim.raw_dim();
        let nelem = dim.size();
        let idx = unsafe { self.reserve(nelem) };
        let xs = unsafe { self.slice_mut(idx, raw_dim) };
        ArrayViewMut::from_shape(dim, xs).unwrap()
    }

    pub fn zeros<'a, D: Dimension, Sh: GDim<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let mut arr = self.empty(dim);
        arr.fill(T::zero());
        arr
    }

    pub fn clone<'a, D: Dimension>(&'a self, xs: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
        let mut empty = self.empty(xs.raw_dim());
        empty.assign(&xs);
        empty
    }

    unsafe fn slice<D: Dimension>(&self, idx: usize, dim: &D) -> &[T] {
        &self.buf[idx..idx+dim.size()]
    }

    unsafe fn slice_mut<D: Dimension>(&self, idx: usize, dim: &D) -> &mut [T] {
        let slice = self.slice(idx, dim);
        let mut_slice = slice as *const [T] as *mut [T];
        &mut *mut_slice
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

impl<T: Float> ArrayCtx<T> for Ctx<T> {
    fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_dimension();
        self.empty::<D, D>(dim)
    }

    fn clear(&mut self) {
        self.buf.clear();
    }
}

impl<T: Float> Ctx<T>
where StandardNormal: Distribution<T> {
    fn randn<'a, D: Dimension, Sh: GDim<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        use rand_distr::{Normal, Distribution};
        let mut arr = self.empty(dim);
        let mut rng = thread_rng();
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        arr.map_mut(|x| {
            *x = normal.sample(&mut rng);
        });
        arr
    }
}

pub trait GDim: nd::ShapeBuilder + Clone {}
impl<T: nd::ShapeBuilder + Clone> GDim for T {}

pub struct Param<'a, T, D> {
    pub w: ArrayViewMut<'a, T, D>,
    pub g: ArrayViewMut<'a, T, D>,
}

impl<'a, T: Float, D: Dimension> Param<'a, T, D>
where StandardNormal: Distribution<T> 
{
    pub fn zeros<Sh: GDim<Dim = D>>(ctx: &'a Ctx<T>, dim: Sh) -> Param<T, D> {
        Param { w: ctx.zeros(dim.clone()), g: ctx.zeros(dim) }
    }

    pub fn randn<Sh: GDim<Dim = D>>(ctx: &'a Ctx<T>, dim: Sh) -> Param<T, D> {
        let w = ctx.randn(dim.clone());
        let g = ctx.zeros(dim);
        Param { w, g }
    }
}

fn view_immut<A, D: Dimension>(view: ArrayViewMut<A, D>) -> ArrayView<A, D> {
    let dim = view.raw_dim();
    let ptr = view.as_ptr();
    unsafe { ArrayView::from_shape_ptr(dim, ptr) }
}

pub struct InstanceNorm<'a, T, D> {
    ci: ArrayView<'a, T, D>,
    inv_sd: ArrayView<'a, T, D>,
    axis: usize,
}


pub fn var_axis<'a, A, D>(ctx: &'a Ctx<A>, x: &ArrayView<A, D>, axis: usize) -> (ArrayViewMut<'a, A, D>, ArrayViewMut<'a, A, D>)
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

pub fn norm_axis<'a, A, D>(ctx: &'a Ctx<A>, x: &ArrayView<A, D>, axis: usize) -> (ArrayView<'a, A, D>, InstanceNorm<'a, A, D>)
where
    A: Float + FromPrimitive,
    D: Dimension + RemoveAxis,
{
    let eps = A::from(1e-6).unwrap();
    let (var, mu) = var_axis(ctx, x, axis);
    
    let inv_sd = var.mapv_into(|x| A::one() / (x + eps).sqrt());
    let mut ci = ctx.clone(x);
    nd::Zip::from(&mut ci).and_broadcast(&mu).and_broadcast(&inv_sd)
        .for_each(|ci, mu, inv_sd| {
            *ci = (*ci - *mu) * *inv_sd;
        });
    let ci = view_immut(ci);
    let inv_sd = view_immut(inv_sd);
    (ci.clone(), InstanceNorm { ci, inv_sd, axis })
}

pub fn dnorm_axis<'a, A, D>(ctx: &'a Ctx<A>, ictx: &InstanceNorm<A, D>, grad: &ArrayView<A, D>) -> ArrayViewMut<'a, A, D>
where
    A: Float,
    D: Dimension + RemoveAxis,
{
    let dot_gi = dot_axis(ctx, &ictx.ci, grad, ictx.axis);

    let mut dy_dc = ctx.empty(grad.raw_dim());
    let axis = Axis(ictx.axis);
    let n = A::from(grad.len_of(axis)).unwrap();

    nd::Zip::from(&ictx.ci).and_broadcast(&ictx.inv_sd).and(grad).and_broadcast(&dot_gi).and(&mut dy_dc)
        .for_each(|ci, inv_sd, grad, dot_gi, dy_dc| {
            *dy_dc = ci.neg() / n * inv_sd.powi(3) * *dot_gi + *grad * *inv_sd;
        });

    
    let dy_dxi_t = mean_axis(ctx,&dy_dc.view(), axis.0);

    nd::Zip::from(&mut dy_dc).and_broadcast(&dy_dxi_t).for_each(|y, x| { *y = *y - *x; } );

    dy_dc
}

pub fn sub<'a, T: Float, D: Dimension>(ctx: &'a Ctx<T>, a: &ArrayView<T, D>, b: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
    let mut buf = ctx.empty(a.raw_dim());

    nd::Zip::from(&mut buf).and(a).and_broadcast(b)
        .for_each(|y, a, b| {
            *y = *a - *b;
        });

    buf
}

pub fn permute<'a, T: Float, D: Dimension, Sh: IntoDimension<Dim = D> + Clone>(ctx: &'a Ctx<T>, a: &ArrayView<T, D>, dim: Sh) -> ArrayViewMut<'a, T, D> {
    let a = a.clone().permuted_axes(dim.clone());
    let mut buf = ctx.empty(a.raw_dim());
    buf.assign(&a);
    buf
}

pub fn matmul<'a, T: Float + 'static>(ctx: &'a Ctx<T>, a: &ArrayView2<T>, b: &ArrayView2<T>) -> ArrayViewMut2<'a, T> {
    let dim_a = a.raw_dim();
    let dim_b = b.raw_dim();
    let mut buf = ctx.empty([dim_a[0], dim_b[1]]);
    general_mat_mul(T::one(), a, b, T::zero(), &mut buf);
    buf
}

pub fn dmatmul<'a, T: Float + 'static>(ctx: &'a Ctx<T>, grad: &ArrayView2<T>, a: &ArrayView2<T>, b: &ArrayView2<T>) -> (ArrayViewMut2<'a, T>, ArrayViewMut2<'a, T>) {
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

pub fn dot_axis<'a, A: Float, D: Dimension + RemoveAxis>(ctx: &'a Ctx<A>, x: &ArrayView<A, D>, y: &ArrayView<A, D>, axis: usize) -> ArrayViewMut<'a, A, D> {
    let mut buf = unit_axis(ctx, x.raw_dim(), axis);
    for (view_a, view_b) in x.axis_iter(Axis(axis)).zip(y.axis_iter(Axis(axis))) {
        for ((z, a), b) in buf.iter_mut().zip(view_a.iter()).zip(view_b.iter()) {
            *z = a.mul_add(*b, *z);
        }
    }
    buf
}

pub fn mean_axis<'a, A: Float, D: Dimension + RemoveAxis>(ctx: &'a Ctx<A>, x: &ArrayView<A, D>, axis: usize) -> ArrayViewMut<'a, A, D> {
    let mut buf = unit_axis(ctx, x.raw_dim(), axis);
    let n = A::from(x.len_of(Axis(axis))).unwrap();
    for view_a in x.axis_iter(Axis(axis)) {
        for (z, a) in buf.iter_mut().zip(view_a.iter()) {
            *z = *z + *a;
        }
    }
    buf.mapv_into(|x| x / n)
}

pub fn unit_axis<'a, D: Dimension, F: Float>(ctx: &'a Ctx<F>, mut dim: D, i: usize) -> ArrayViewMut<'a, F, D> {
    dim.slice_mut()[i] = 1;
    ctx.zeros(dim)
}


pub struct SimpleLinearNode<'a> {
    w: Param<'a, f32, Ix2>,
    b: Param<'a, f32, Ix2>,
    ictx: Option<InstanceNorm<'a, f32, Ix2>>,
    norm_x: Option<ArrayView<'a, f32, Ix2>>,
    y: Option<ArrayView<'a, f32, Ix2>>
}

impl<'a> SimpleLinearNode<'a> {
    pub fn new(ctx: &'a Ctx<f32>, dim: usize) -> Self {
        SimpleLinearNode { 
            w: Param::randn(ctx, [dim, dim]), 
            b: Param::zeros(ctx, [1, dim]), 
            ictx: None,
            norm_x: None,
            y: None,
        }
    }

    pub fn forward(&mut self, ctx: &'a Ctx<f32>, x: ArrayView<f32, Ix2>) -> ArrayView<'a, f32, Ix2> {
        let (norm_x, ictx) = norm_axis(ctx, &x, 1);
        self.ictx = Some(ictx);
        self.norm_x = Some(norm_x.clone());
        let mut y = matmul(ctx, &norm_x, &self.w.w.view());
        y += &self.b.w;
        y.mapv_inplace(|x| {
            crate::ops::relu(x)
        });
        let y = view_immut(y);
        self.y = Some(y.clone());
        y
    }

    pub fn backward(&mut self, ctx: &'a Ctx<f32>, grad: ArrayView<f32, Ix2>) -> Option<ArrayViewMut<'a, f32, Ix2>> {
        if self.ictx.is_none() || self.y.is_none() || self.norm_x.is_none() {
            return None;
        }
        // dl/drelu
        let dy = std::mem::replace(&mut self.y, None).unwrap();
        let mut dy = ctx.clone(&dy);
        dy.zip_mut_with(&grad, |y, g| {
            if *y > 0.0 {
                *y = *g;
            } else {
                *y = 0.0;
            }
        }); // now dy contains dl/dy

        // compute dy_db by summing over broadcasting dim 0
        let db = &mut self.b.g;
        for slice in dy.axis_iter(Axis(0)) { // this prevents extra allocation
            db.zip_mut_with(&slice, |db, dy| {
                *db += *dy;
            });
        }

        let norm_x = std::mem::replace(&mut self.norm_x, None).unwrap();

        let (dnorm_x, dw) = dmatmul(ctx, &dy.view(), &norm_x, &self.w.w.view());
        self.w.g += &dw;

        // finally compute the derivate w.r.t. the input
        let ictx = std::mem::replace(&mut self.ictx, None).unwrap();
        let dx = dnorm_axis(ctx, &ictx, &dnorm_x.view());
        Some(dx)
    }
}


pub struct SimpleGrid<'a> {
    grid: Vec<SimpleLinearNode<'a>>, // shape: [z, y, x]
    value_grid1: ArrayViewMut3<'a, f32>,     // shape: [y * x, b, d]
    value_grid2: ArrayViewMut3<'a, f32>,
    xyz: [usize; 3],
}

impl<'a> SimpleGrid<'a> {
    pub fn new(ctx: &'a Ctx<f32>, xyz: [usize; 3], batch_size: usize, dim: usize) -> Self {
        let nelem = xyz[0] * xyz[1] * xyz[2];
        let value_grid1 = ctx.zeros((xyz[1] * xyz[0], batch_size, dim));
        let value_grid2 = ctx.zeros((xyz[1] * xyz[0], batch_size, dim));
        let grid = (0..nelem).map(|_| { SimpleLinearNode::new(ctx, dim) }).collect();
        Self { grid, value_grid1, value_grid2, xyz }
    }

    // expect x is of shape [b, d, y, x]
    pub fn forward(&mut self, ctx: &'a Ctx<f32>, xs: &ArrayView4<f32>) -> ArrayView4<'a, f32> {
        let (b, d, y, x) = xs.dim();
        let xs = xs.into_shape((b, d, y * x)).unwrap();
        let xs = permute(ctx, &xs, [2, 0, 1]);
        // permute and copy into value grid, so that the shape is [y * x, b, d]
        self.value_grid1.zip_mut_with(&xs, |v, x| {
            *v = *x;
        });
        let xy_stride = self.xyz[0] * self.xyz[1];
        for z in 0..self.xyz[2] {
            // propagate one layer through
            let zslice = &mut self.grid[z*xy_stride..(z+1)*xy_stride];
            for (i, mut result) in self.value_grid1.axis_iter_mut(Axis(0)).enumerate() {
                // result is of shape [b, d]
                let output = zslice[i].forward(ctx, result.view());
                result.assign(&output);
            }

            // now aggregate the results
            for iy in 0..self.xyz[1] {
                for ix in 0..self.xyz[0] {
                    let xstride = self.xyz[0];
                    let ic = iy * xstride + ix;
                    let mut acc_slice = self.value_grid2.index_axis_mut(Axis(0), ic);
                    acc_slice.fill(0.0);
                    for ky in -1..=1 { // average sliding window with kernel size 3, pad 1 and stride 1
                        for kx in -1..=1 {
                            let ikx = ix as isize + kx;
                            let iky = iy as isize + ky;
                            if ikx >= 0 && ikx < self.xyz[0] as isize && iky >= 0 && iky < self.xyz[1] as isize {
                                let ik = ikx as usize + iky as usize * xstride;
                                acc_slice += &self.value_grid1.index_axis(Axis(0), ik);
                            }
                        }
                    }
                }
            }
            std::mem::swap(&mut self.value_grid1, &mut self.value_grid2);
        }
        // now the output should be in self.value_grid1
        let output = permute(ctx, &self.value_grid1.view(), [1, 2, 0]);
        let output = output.into_shape([b, d, y, x]).unwrap();
        view_immut(output)
    }

}

#[test]
fn test_simple_compute_grid() {
    let ctx = Ctx::new(8 * 16 * 16 * 16 * 10);
    let mut grid = SimpleGrid::new(&ctx, [5, 5, 5], 1, 2);
    let input = ctx.randn((1, 2, 5, 5));
    let _output = grid.forward(&ctx, &input.view());
    println!("{}", ctx.buf.len());
}
