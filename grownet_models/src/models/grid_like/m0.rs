use ndarray::prelude::*;
use ndarray as nd;
use num::Float;
use rand::thread_rng;

use crate::nn::nd_ops::{*, ops_owned as ops};
use crate::ops::relu;


pub struct SimpleLinearNode {
    w: Param<f32, Ix2>,
    b: Param<f32, Ix2>,
    ictx: Option<ops::InstanceNorm<f32, Ix2>>,
    norm_x: Option<Array<f32, Ix2>>,
    y: Option<Array<f32, Ix2>>
}

impl SimpleLinearNode {
    pub fn new(dim: usize) -> Self {
        SimpleLinearNode { 
            w: Param::randn([dim, dim]), 
            b: Param::zeros([1, dim]), 
            ictx: None,
            norm_x: None,
            y: None,
        }
    }

    pub fn forward(&mut self, x: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
        let (norm_x, ctx) = ops::norm_axis(&x, 1);
        self.ictx = Some(ctx);
        self.norm_x = Some(norm_x.clone());
        let mut y = norm_x.dot(&self.w.w);
        y += &self.b.w;
        y.mapv_inplace(|x| {
            relu(x)
        });
        self.y = Some(y.clone());
        y
    }

    pub fn backward(&mut self, grad: ArrayView<f32, Ix2>) -> Option<Array<f32, Ix2>> {
        if self.ictx.is_none() || self.y.is_none() || self.norm_x.is_none() {
            return None;
        }
        // dl/drelu
        let mut dy = std::mem::replace(&mut self.y, None).unwrap();
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

        let (dnorm_x, dw) = ops::dmatmul(&dy, &norm_x, &self.w.w);
        self.w.g += &dw;

        // finally compute the derivate w.r.t. the input
        let ictx = std::mem::replace(&mut self.ictx, None).unwrap();
        let dx = ops::dnorm_axis(&ictx, &dnorm_x.view());
        Some(dx)
    }
}

pub struct SimpleGrid {
    grid: Vec<SimpleLinearNode>, // shape: [z, y, x]
    value_grid1: Array3<f32>,     // shape: [y * x, b, d]
    value_grid2: Array3<f32>,
    xyz: [usize; 3],
}

impl SimpleGrid {
    pub fn new(xyz: [usize; 3], batch_size: usize, dim: usize) -> Self {
        let nelem = xyz[0] * xyz[1] * xyz[2];
        let value_grid = Array3::zeros((xyz[1] * xyz[0], batch_size, dim));
        let grid = (0..nelem).map(|_| { SimpleLinearNode::new(dim) }).collect();
        Self { grid, value_grid1: value_grid.clone(), value_grid2: value_grid, xyz }
    }

    // expect x is of shape [y, x, b, d]
    pub fn forward(&mut self, xs: &ArrayView4<f32>) -> ArrayView4<f32> {
        let (y, x, b, d) = xs.dim();
        let xs = xs.into_shape((y * x, b, d)).expect("incorrect input shape");
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
                let output = zslice[i].forward(result.view());
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
        let output = self.value_grid1.view().into_shape([y, x, b, d]).unwrap();
        output
    }

}




#[test]
fn test_linear_node() {
    let mut node = SimpleLinearNode::new(16);
    let x = ops::randn((4, 16));
    let _y = node.forward(x.view());
    let g = ops::randn((4, 16));
    let _dy = node.backward(g.view());
}

#[test]
fn test_simple_compute_grid() {
    let mut grid = SimpleGrid::new([16, 16, 16], 1, 8);
    let input = ops::randn((16, 16, 1, 8));
    let _output = grid.forward(&input.view());
}

