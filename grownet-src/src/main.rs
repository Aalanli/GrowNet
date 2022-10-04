#![allow(dead_code)]

mod m2;
mod tensor;

use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use num::traits::float::FloatCore;
use tensor as ts;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal, Distribution};

#[derive(Clone, Copy)]
struct Test(f32);

impl Default for Test {
    fn default() -> Self {
        Test(1.0)
    }
}

fn main() {
}
