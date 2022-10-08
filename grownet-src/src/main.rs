#![allow(dead_code)]

mod m2;
mod tensor;

use std::marker::PhantomData;
use std::process::Output;
use std::fmt::Display;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use num::traits::float::FloatCore;
use tensor as ts;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal, Distribution};
use std::ptr;
use std::mem;

fn main() {
    let k: *mut Vec<usize>;
    unsafe {
        let v = vec![1usize, 2, 3];
        let mut v = mem::ManuallyDrop::new(v);
        k = &mut *v as *mut Vec<usize>;
        println!("val {}", (*k)[0]);
        ptr::drop_in_place(k);
    }
}
