use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};

/// Single threaded buffer holding a large number of various shaped tensors of different dimensions in
/// contiguous memory
struct TensorCtx<T, const N: usize> {
    workbuf: *mut T,
    ptr: NonNull<T>,
    dims: *mut [usize; N],
    strides: *mut [usize; N],
    offsets: *mut usize,
    len: usize,
    alloc_elems: usize
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