use std::ptr::NonNull;
use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr;
use std::ops::{Index, IndexMut};
use std::iter::Iterator;

use super::TIndex;


pub enum TensorSlice {
    Edge,
    Index(usize)
}

pub struct SliceSeq(Vec<(TensorSlice, TensorSlice)>);



pub struct WorldTensor<T> {
    ptr: NonNull<T>,         // NonNull pointer
    dims: Vec<usize>,        // dimensions row-major
    strides: Vec<usize>,     // strides row-major
    len: usize,              // total n elements
    marker: PhantomData<T>,  // tells compiler that Tensor owns values of type T
}

pub enum TensorSlice {
    
}

pub struct WorldSlice<'a, T> {
    world: &'a WorldTensor<T>,
    trunc_dims: Vec<(usize, usize)>,
}

pub struct WorldMutSlice<'a, T> {
    world: &'a mut WorldTensor<T>,
    trunc_dims: Vec<(usize, usize)>,
}

impl<T> WorldTensor<T> 
{
    pub fn new(dims: Vec<usize>) -> WorldTensor<T> {
        let nelems = dims.iter().fold(1, |a, b| a * b);
        assert!(nelems > 0, "cannot make 0 sized tensor");

        let layout = Layout::array::<T>(nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };

        let mut strides = vec![];
        let mut k = 1usize;
        for i in (0..dims.len()).rev() {
            strides.push(k);
            k *= dims[i];
        }

        WorldTensor{ptr, dims, strides, len: nelems, marker: PhantomData}
    }

}

impl<T> Drop for WorldTensor<T> {
    fn drop(&mut self) {
        for i in 0..self.len {
            unsafe{ ptr::drop_in_place(self.ptr.as_ptr().add(i)); }
        }
        let layout = Layout::array::<T>(self.len).unwrap();
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T, I> Index<I> for WorldTensor<T> 
where I: TIndex
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        if let Some(idx) = ind.tile_cartesian(&self.dims, &self.strides) {
            return inbounds_get_ptr(&self.ptr, idx);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, I> IndexMut<I> for WorldTensor<T> 
where I: TIndex
{
    fn index_mut(&mut self, ind: I) -> &mut T {
        if let Some(idx) = ind.tile_cartesian(&self.dims, &self.strides) {
            return inbounds_get_ptr_mut(&mut self.ptr, idx);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T> WorldSlice<'a, T> {
    //fn new(tensor: &'a WorldTensor<T>) -> WorldSlice<'a, T> {
//
    //}
}

fn inbounds_get_ptr<T>(ptr: &NonNull<T>, i: usize) -> &T {
    unsafe{ ptr.as_ptr().add(i).as_ref().unwrap() }
}

fn inbounds_get_ptr_mut<T>(ptr: &mut NonNull<T>, i: usize) -> &mut T {
    unsafe { ptr.as_ptr().add(i).as_mut().unwrap() }
}

fn set_ptr<T>(ptr: &mut NonNull<T>, val: T, i: usize) {
    unsafe {
        let raw_ptr = ptr.as_ptr().add(i);
        ptr::drop_in_place(raw_ptr);
        ptr::write(raw_ptr, val);
    }
}

fn grow_dim_<T>(ptr: NonNull<T>, size: usize, rchunk: usize, lchunk: usize, fsize: usize, bsize: usize) -> (NonNull<T>, usize) {
    let chunksz = rchunk + fsize + bsize;
    let new_len = lchunk * chunksz;
    let old_layout = Layout::array::<T>(size).unwrap();
    let new_layout = Layout::array::<T>(new_len).unwrap();
    let new_ptr = {
        let old_ptr = ptr.as_ptr() as *mut u8;
        let new_ptr = unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) as *mut T };

        // initiate copying operations to move each dimension to its proper place
        unsafe {
            if bsize > 0 {
                ptr::copy(new_ptr, new_ptr.add(bsize), bsize);
            }
            for i in (1..lchunk).rev() {
                ptr::copy_nonoverlapping(new_ptr.add(
                    i * rchunk), new_ptr.add(i * chunksz + bsize), rchunk);
            }
        };
        
        new_ptr
    };

    let ptr = match NonNull::new(new_ptr) {
        Some(p) => p,
        None  => alloc::handle_alloc_error(new_layout)
    };

    (ptr, new_len)
}