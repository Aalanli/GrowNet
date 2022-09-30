use std::ptr::NonNull;
use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr;
use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::iter::Iterator;

use super::index::{TIndex, UnsafeIndex};
use super::slice::{slice, IndexBounds, TsSlices, Slice, Edge, S};

trait Tensor<T, I>: Index<I, Output = T> {
    fn slice(&self, slices: TsSlices) -> WorldSlice<'_, T>;
    fn clone(&self) -> Self;
    fn iter(&self) -> TsIter<T>;
    fn nelems(&self) -> usize;
    fn ptr(&self) -> *const T;
    fn ptr_mut(&mut self) -> *mut T;
}

trait MutTensor<T, I>: IndexMut<I, Output = T> + Tensor<T, I> {
    fn mut_slice(&mut self, slices: TsSlices) -> MutWorldSlice<'_, T>;
}

pub struct WorldTensor<T> {
    ptr: NonNull<T>,         // NonNull pointer
    dims: Vec<usize>,        // dimensions row-major
    strides: Vec<usize>,     // strides row-major
    len: usize,              // total n elements
    marker: PhantomData<T>,  // tells compiler that Tensor owns values of type T
}


// WorldSlice represents a slice of a WorldTensor,
// 
// params: 
// trunc_dims - a pair representing (offset, size) of the slice of a dimension
// dims - representing the number of slices with size greater than 1
pub struct WorldSlice<'a, T> {
    world: &'a WorldTensor<T>,
    slice: Slice
}

pub struct MutWorldSlice<'a, T> {
    world: &'a mut WorldTensor<T>,
    slice: Slice
}

pub struct TsIter<'a, T> {
    world: &'a T,
    ind: usize
}

pub struct MutTsIter<'a, T> {
    world: &'a mut T,
    ind: usize
}


impl<T> WorldTensor<T> 
{
    pub fn new(dims: Vec<usize>) -> WorldTensor<T> {
        let strides= compute_strides(&dims);
        let nelems = strides[0] * dims[0];
        assert!(nelems > 0, "cannot make 0 sized tensor");

        let layout = Layout::array::<T>(nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };


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

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![];
    strides.reserve_exact(dims.len());
    let mut k = 1usize;
    for i in (0..dims.len()).rev() {
        strides.push(k);
        k *= dims[i];
    }
    strides
}

impl<'a, T: Tensor<T, usize>> Iterator for TsIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < self.world.nelems() {
            self.ind += 1;
            Some(&self.world[self.ind])
        } else {
            None
        }
    }
}

impl<'a, T> Iterator for MutTsIter<'a, MutWorldSlice<'a, T>> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < self.world.world.len {
            self.ind += 1;
            unsafe {
                let i = tile_strides(self.ind, &self.world.slice.strides, 
                    &self.world.world.strides);
                Some(self.world.world.ptr.as_ptr().add(i).as_mut().unwrap())
            }
        } else {
            None
        }
    }
}

// Have to watchout for the case where the size of the slice is 0, 
// as that state represents the removal of a dimension through slicing
// similar to the case i:i+1, but that case preserves the dimension
fn construct_slice(dims: &[usize], strides: &[usize], slices: TsSlices) -> Slice {
    let mut offsets = Vec::<usize>::new();
    let mut sizes = Vec::<usize>::new();

    let mut new_dims = dims.len();
    let mut lin_offset = 0usize;

    let wrap = |x: isize, d, i| {
        if x.abs() as usize > d {
            panic!("pos {} of slice {} is out of bounds", slices[i], slices);
        }
        if x < 0 {d + x as usize}
        else {x as usize}
    };

    for (s, i) in (*slices).iter().zip(0..slices.len()) {
        let d = dims[i];
        match s {
            IndexBounds(Edge, Edge) => {offsets.push(0); sizes.push(d)},
            IndexBounds(S(l), Edge) => {
                let e = wrap(*l, d, i);
                offsets.push(e);
                sizes.push(d-e);
            },
            IndexBounds(Edge, S(r)) => {
                let e = wrap(*r, d, i);
                offsets.push(0);
                sizes.push(e);
            },
            IndexBounds(S(l), S(r)) => {
                if l > r {
                    panic!("left index {} is greater than right {} index for slice {}", l, r, slices);
                }
                let e1 = wrap(*l, d, i);

                if l == r {
                    new_dims -= 1;
                    lin_offset += strides[i] * e1;
                } else {
                    offsets.push(e1);
                    sizes.push(wrap(*r, d, i) - e1);
                }
            }
        };
    }
    for j in slices.len()..dims.len() {
        offsets.push(0);
        sizes.push(dims[j]);
    }
    let strides = compute_strides(&offsets);

    Slice{offsets, sizes, non_zero_dims: new_dims, lin_offset, strides}
}

impl<'a, T> WorldSlice<'a, T> {
    fn new(tensor: &'a WorldTensor<T>, slices: TsSlices) -> WorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices);
        WorldSlice { world: &tensor, slice }
    }
}

impl<'a, T> MutWorldSlice<'a, T> {
    fn new(tensor: &'a mut WorldTensor<T>, slices: TsSlices) -> MutWorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices);
        MutWorldSlice { world: tensor, slice }
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


impl<T> Index<usize> for WorldTensor<T> 
{
    type Output = T;
    fn index(&self, ind: usize) -> &T {
        if ind > self.len {
            return inbounds_get_ptr(&self.ptr, ind);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T> Index<UnsafeIndex<usize>> for WorldTensor<T> 
{
    type Output = T;
    fn index(&self, ind: UnsafeIndex<usize>) -> &T {
        return inbounds_get_ptr(&self.ptr, *ind);
    }
}

impl<T> IndexMut<usize> for WorldTensor<T> 
{
    fn index_mut(&mut self, ind: usize) -> &mut T {
        if ind > self.len {
            return inbounds_get_ptr_mut(&mut self.ptr, ind);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T> IndexMut<UnsafeIndex<usize>> for WorldTensor<T> 
{
    fn index_mut(&mut self, ind: UnsafeIndex<usize>) -> &mut T {
        return inbounds_get_ptr_mut(&mut self.ptr, *ind);
    }
}


impl<'a, T, I> Index<I> for WorldSlice<'a, T> 
where I: TIndex
{
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        if let Some(idx) = index.offset(&self.slice.offsets).tile_cartesian(&self.slice.sizes, &self.world.strides) {
            return inbounds_get_ptr(&self.world.ptr, idx + self.slice.lin_offset);
        } else {
            panic!("Index out of bounds");
        }
    }
}

fn tile_strides(index: usize, s_strides: &[usize], g_strides: &[usize]) -> usize{
    let mut idx = 0;
    let mut ind = index;
    for i in 0..s_strides.len() {
        let c = ind / s_strides[i] as usize;
        idx += c * g_strides[i];
        ind -= c * s_strides[i];
        if ind <= 0 {
            break;
        }
    }
    idx
}

impl<'a, T> Index<usize> for WorldSlice<'a, T>
{
    type Output = T;
    fn index(&self, ind: usize) -> &T {
        let ind = tile_strides(ind, &self.slice.strides, &self.world.strides);
        if ind > self.world.len {
            return inbounds_get_ptr(&self.world.ptr, ind);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T> Index<UnsafeIndex<usize>> for WorldSlice<'a, T>
{
    type Output = T;
    fn index(&self, ind: UnsafeIndex<usize>) -> &T {
        let ind = tile_strides(*ind, &self.slice.strides, &self.world.strides);
        return inbounds_get_ptr(&self.world.ptr, ind);
    }
}

impl<'a, T> Index<usize> for MutWorldSlice<'a, T>
{
    type Output = T;
    fn index(&self, ind: usize) -> &T {
        let ind = tile_strides(ind, &self.slice.strides, &self.world.strides);
        if ind > self.world.len {
            return inbounds_get_ptr(&self.world.ptr, ind);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T> Index<UnsafeIndex<usize>> for MutWorldSlice<'a, T>
{
    type Output = T;
    fn index(&self, ind: UnsafeIndex<usize>) -> &T {
        let ind = tile_strides(*ind, &self.slice.strides, &self.world.strides);
        return inbounds_get_ptr(&self.world.ptr, ind);
    }
}

impl<'a, T> IndexMut<usize> for MutWorldSlice<'a, T>
{
    fn index_mut(&mut self, ind: usize) -> &mut T {
        let ind = tile_strides(ind, &self.slice.strides, &self.world.strides);
        if ind > self.world.len {
            return inbounds_get_ptr_mut(&mut self.world.ptr, ind);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T> IndexMut<UnsafeIndex<usize>> for MutWorldSlice<'a, T> 
{
    fn index_mut(&mut self, ind: UnsafeIndex<usize>) -> &mut T {
        let ind = tile_strides(*ind, &self.slice.strides, &self.world.strides);
        return inbounds_get_ptr_mut(&mut self.world.ptr, ind);
    }
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