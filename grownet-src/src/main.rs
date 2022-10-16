#![allow(dead_code)]

mod m2;
mod tensor;
use std::{marker::PhantomData, ops::Index};

use std::process::Output;
use std::fmt::Display;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use num::complex::ComplexFloat;
use num::traits::float::FloatCore;
use tensor as ts;
use ndarray::prelude::*;
use ndarray::{Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal, Distribution};
use std::ptr;
use std::mem;

fn main() {
    let mut a: Array<f32, _> = Array::random((512, 512), Normal::new(0.0, 1.0).unwrap());
    let b: Array<f32, _> = Array::random((512, 512), Normal::new(0.0, 1.0).unwrap());
    let c = &a - &b;
    a.zip_mut_with(&b, |x, y| {*x -= y;} );

    let max = c.iter().zip(a.iter()).fold(-10000.0f32, |c, (x, y)| {c.max((x-y).abs())});
    println!("{max}");
}
