use std::marker::PhantomData;

use nd::IntoDimension;
use nd::linalg::general_mat_mul;
use ndarray::prelude::*;
use ndarray::{self as nd, RemoveAxis};
use ndarray_rand::{rand, rand::thread_rng, rand_distr::{Normal, Distribution, StandardNormal}, RandomExt};

use num::{Float, FromPrimitive};
use crate::nn::nd_ops as ops;
use crate::nn::nd_ops::context::{FlatCtx, NaiveCtx};
use ops::ops_ctx::*;


pub struct SimpleLinearNode {
    w: ops::Param<f32, Ix2>,
    b: ops::Param<f32, Ix2>,
    ictx: Option<InstanceNorm<f32, Ix2>>,
    norm_x: Option<ArrId<f32, Ix2>>,
    y: Option<ArrId<f32, Ix2>>
}

impl SimpleLinearNode {
    pub fn new(dim: usize) -> Self {
        SimpleLinearNode { 
            w: ops::Param::randn([dim, dim]), 
            b: ops::Param::zeros([1, dim]), 
            ictx: None,
            norm_x: None,
            y: None,
        }
    }

    pub fn forward<'a, Ctx: ArrayCtx<f32>>(&mut self, ctx: &'a Ctx, x: ArrayView<f32, Ix2>) -> ArrayView<'a, f32, Ix2> {
        let (norm_x, ictx) = norm_axis(ctx, &x, 1);
        self.ictx = Some(ictx);
        let mut y = matmul(ctx, &norm_x.view(), &self.w.w.view());
        self.norm_x = Some(ctx.id(norm_x));
        y += &self.b.w;
        y.mapv_inplace(|x| {
            crate::ops::relu(x)
        });
        
        let y = view_immut(y);
        self.y = Some(ctx.id(ctx.clone(&y)));
        y
    }

    pub fn backward<'a, Ctx: ArrayCtx<f32>>(&mut self, ctx: &'a Ctx, grad: ArrayView<f32, Ix2>) -> Option<ArrayViewMut<'a, f32, Ix2>> {
        if self.ictx.is_none() || self.y.is_none() || self.norm_x.is_none() {
            return None;
        }
        // dl/drelu
        let dy = std::mem::replace(&mut self.y, None).unwrap();
        let mut dy = ctx.clone(&ctx.from_id(&dy));
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
        let norm_x = ctx.from_id(&norm_x);
        let (dnorm_x, dw) = dmatmul(ctx, &dy.view(), &norm_x, &self.w.w.view());
        self.w.g += &dw;

        // finally compute the derivate w.r.t. the input
        let ictx = std::mem::replace(&mut self.ictx, None).unwrap();
        let dx = dnorm_axis(ctx, &ictx, &dnorm_x.view());
        Some(dx)
    }
}


pub struct SimpleGrid {
    grid: Vec<SimpleLinearNode>, // shape: [z, y, x]
    xyz: [usize; 3],
}

impl SimpleGrid {
    pub fn new(xyz: [usize; 3], dim: usize) -> Self {
        let nelem = xyz[0] * xyz[1] * xyz[2];
        let grid = (0..nelem).map(|_| { SimpleLinearNode::new(dim) }).collect();
        Self { grid, xyz }
    }

    // expect x is of shape [y, x, b, d]
    pub fn forward<'a, Ctx: ArrayCtx<f32>>(&mut self, ctx: &'a Ctx, xs: &ArrayView4<f32>) -> ArrayViewMut4<'a, f32> {
        
        let (y, x, b, d) = xs.dim();
        let mut value_grid2 = ctx.zeros((self.xyz[1] * self.xyz[0], b, d));
        let mut value_grid1 = ctx.zeros((self.xyz[1] * self.xyz[0], b, d));
        let xs = xs.into_shape((y * x, b, d)).expect("incorrect input shape");
        // permute and copy into value grid, so that the shape is [y * x, b, d]
        value_grid1.zip_mut_with(&xs, |v, x| {
            *v = *x;
        });
        let xy_stride = self.xyz[0] * self.xyz[1];
        for z in 0..self.xyz[2] {
            // propagate one layer through
            let zslice = &mut self.grid[z*xy_stride..(z+1)*xy_stride];
            for (i, mut result) in value_grid1.axis_iter_mut(Axis(0)).enumerate() {
                // result is of shape [b, d]
                let output = zslice[i].forward(ctx, result.view());
                result.assign(&output);
            }

            // now aggregate the results
            for iy in 0..self.xyz[1] {
                for ix in 0..self.xyz[0] {
                    let xstride = self.xyz[0];
                    let ic = iy * xstride + ix;
                    let mut acc_slice = value_grid2.index_axis_mut(Axis(0), ic);
                    acc_slice.fill(0.0);
                    for ky in -1..=1 { // average sliding window with kernel size 3, pad 1 and stride 1
                        for kx in -1..=1 {
                            let ikx = ix as isize + kx;
                            let iky = iy as isize + ky;
                            if ikx >= 0 && ikx < self.xyz[0] as isize && iky >= 0 && iky < self.xyz[1] as isize {
                                let ik = ikx as usize + iky as usize * xstride;
                                acc_slice += &value_grid1.index_axis(Axis(0), ik);
                            }
                        }
                    }
                }
            }
            std::mem::swap(&mut value_grid1, &mut value_grid2);
        }
        // now the output should be in self.value_grid1
        let output = value_grid1.into_shape([y, x, b, d]).unwrap();
        output
    }

}

#[test]
fn test_simple_compute_grid() {
    let ctx = FlatCtx::<f32>::new(8 * 16 * 16 * 16 * 10);
    let mut grid = SimpleGrid::new([5, 5, 5], 2);
    let input = randn(&ctx, (5, 5, 1, 2));
    let _output = grid.forward(&ctx, &input.view());
}

#[test]
fn test_mem_usage() {
    let ctx = NaiveCtx::<f32>::new();
    let mut grid = SimpleGrid::new([32, 32, 32], 8);
    let input = randn(&ctx, (32, 32, 1, 8));
    let _output = grid.forward(&ctx, &input.view());
    println!("used {}", ctx.allocated());
}