use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};

use ndarray::prelude::*;

/// Single threaded buffer holding a large number of tensors of the same dimension
pub struct TensorCtx<T, const N: usize> {
    workbuf: NonNull<T>, // temporary accumulator buffer
    ptr: NonNull<T>, // buffer holding flattened tensors
    dims: *mut [usize; N],
    strides: *mut [usize; N],
    offsets: *mut usize,
    len: usize,
    alloc_elems: usize
}

impl<T, const N: usize> TensorCtx<T, N> {
    fn alloc(ptr: NonNull<T>) {
        
    }
    pub fn new() -> Self {
        todo!()
    }
}

fn transpose_slice<T: Copy, const N: usize>(
    slice: &mut [T], 
    i_stride: usize, i_dim: usize,
    j_stride: usize, j_dim: usize) 
{
    let ptr = slice.as_ptr() as *mut T;
    //let len = slice.len();
    
    unsafe {
        for i in 1..i_dim {
            for j in (i+1)..j_dim {
                let it = i * i_stride + j * j_stride;
                let jt = j * i_stride + i * j_stride;
                let temp = *ptr.add(it);
                *ptr.add(it) = *ptr.add(jt);
                *ptr.add(jt) = temp;
            }
        }
    }
}