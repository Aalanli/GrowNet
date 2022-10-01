use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::ops::{Index, IndexMut};
use std::iter::Iterator;
use std::convert::{From, Into};
use std::fmt::Display;

use super::index::{self as ind, TsIndex, TileIndex, IndexReg, IndexSlice, tile_strides};
use super::slice::{IndexBounds, TsSlices, Slice, Edge, S};

trait Tensor<T, I>: Index<I, Output = T> + Sized {
    fn slice(&self, slices: TsSlices) -> WorldSlice<'_, T>;
    fn iter(&self) -> TsIter<Self>;
    fn nelems(&self) -> usize;
    fn ptr(&self) -> *const T;
    fn lin_index(&self, idx: usize) -> usize;
}

trait MutTensor<T, I>: IndexMut<I, Output = T> + Tensor<T, I> {
    fn mut_slice(&mut self, slices: TsSlices) -> MutWorldSlice<'_, T>;
    fn ptr_mut(&mut self) -> *mut T;
    fn iter_mut(&mut self) -> MutTsIter<Self>;
}

pub struct WorldTensor<T> {
    pub ptr: NonNull<T>,         // NonNull pointer
    pub dims: Vec<usize>,        // dimensions row-major
    pub strides: Vec<usize>,     // strides row-major
    pub len: usize,              // total n elements
    pub marker: PhantomData<T>,  // tells compiler that Tensor owns values of type T
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

impl<T, I: TsIndex> Tensor<T, I> for WorldTensor<T>
where <<I as TsIndex>::IndexT as TileIndex>::Meta: From<*const WorldTensor<T>> 
{
    fn slice(&self, slices: TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice(&self.dims, &self.strides, slices);
        WorldSlice { world: self, slice }
    }
    fn iter(&self) -> TsIter<Self> {
        TsIter { world: self, ind: 0 }
    }
    fn nelems(&self) -> usize {
        self.len
    }
    fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
    fn lin_index(&self, idx: usize) -> usize {
        idx
    }
}

impl<T, I: TsIndex> MutTensor<T, I> for WorldTensor<T> 
where <<I as TsIndex>::IndexT as TileIndex>::Meta: From<*const WorldTensor<T>> 
{
    fn iter_mut(&mut self) -> MutTsIter<Self> {
        MutTsIter { world: self, ind: 0 }
    }
    fn mut_slice(&mut self, slices: TsSlices) -> MutWorldSlice<'_, T> {
        let slice = construct_slice(&self.dims, &self.strides, slices);
        MutWorldSlice { world: self, slice }
    }
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'a, T, I: TsIndex> Tensor<T, I> for WorldSlice<'a, T>
where ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const WorldSlice<'a, T>>
{
    fn slice(&self, slices: TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice.sizes, &self.slice.offsets, 
            &self.slice.ref_inds, &self.world.strides, 
            self.slice.lin_offset, slices);
        WorldSlice { world: self.world, slice }
    }
    fn iter(&self) -> TsIter<Self> {
        TsIter { world: self, ind: 0 }
    }
    fn nelems(&self) -> usize {
        self.world.len
    }
    fn ptr(&self) -> *const T {
        self.world.ptr.as_ptr()
    }
    fn lin_index(&self, idx: usize) -> usize {
        unsafe {
            tile_strides(idx, &(self as *const Self).into())
        }
    }
}


impl<'a, T, I: TsIndex> Tensor<T, I> for MutWorldSlice<'a, T>
where ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const MutWorldSlice<'a, T>>
{
    fn slice(&self, slices: TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice.sizes, &self.slice.offsets, 
            &self.slice.ref_inds, &self.world.strides, 
            self.slice.lin_offset, slices);
        WorldSlice { world: self.world, slice }
    }
    fn iter(&self) -> TsIter<Self> {
        TsIter { world: self, ind: 0 }
    }
    fn nelems(&self) -> usize {
        self.world.len
    }
    fn ptr(&self) -> *const T {
        self.world.ptr.as_ptr()
    }
    fn lin_index(&self, idx: usize) -> usize {
        unsafe {
            tile_strides(idx, &(self as *const Self).into())
        }
    }
}

impl<'a, T, I: TsIndex> MutTensor<T, I> for MutWorldSlice<'a, T> 
where ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const MutWorldSlice<'a, T>>
{
    fn iter_mut(&mut self) -> MutTsIter<Self> {
        MutTsIter { world: self, ind: 0 }
    }
    fn mut_slice(&mut self, slices: TsSlices) -> MutWorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice.sizes, &self.slice.offsets, 
            &self.slice.ref_inds, &self.world.strides, 
            self.slice.lin_offset, slices);
        MutWorldSlice { world: self.world, slice }
    }
    fn ptr_mut(&mut self) -> *mut T {
        self.world.ptr.as_ptr()
    }
}


impl<T> WorldTensor<T> {
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

impl<T: Display> Display for WorldTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        recursive_write(f, self.ptr.as_ptr(), 0, 0, &self.dims, &self.strides, self.len);
        write!(f, "")
    }
}

fn recursive_write<T: Display>(f: &mut std::fmt::Formatter<'_>, arr: *const T, cur_dim: usize, cur_idx: usize, dim: &[usize], strides: &[usize], n_elems: usize) {
    if cur_dim == dim.len() - 1 {
        write!(f, "[").unwrap();
        for i in 0..dim[cur_dim]-1 {
            if cur_idx + i < n_elems {
                let v = unsafe {&*arr.add(i + cur_idx)};
                write!(f, "{},", v).unwrap();
            } else {
                panic!("index out of bounds while printing");
            }
        }
        let i = dim[cur_dim] - 1;
        if cur_idx + i < n_elems {
            let v = unsafe {&*arr.add(i + cur_idx)};
            write!(f, "{}]", v).unwrap();
        } else {
            panic!("index out of bounds while printing");
        }
    } else {
        //write!(f, "[\n").unwrap();
        for i in 0..dim[cur_dim]-1 {
            recursive_write(f, arr, cur_dim + 1, cur_idx + i * strides[cur_dim], dim, strides, n_elems);
            write!(f, ",\n").unwrap();
        }
        let i = dim[cur_dim]-1;
        recursive_write(f, arr, cur_dim + 1, cur_idx + i * strides[cur_dim], dim, strides, n_elems);    
        write!(f, "\n").unwrap();
    }
}

impl<T: Clone> Clone for WorldTensor<T> {
    fn clone(&self) -> Self {
        let layout = Layout::array::<T>(self.len).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };
        for i in 0..self.len {
            unsafe {
                *ptr.as_ptr().add(i) = (*self.ptr.as_ptr().add(i)).clone();
            }
        }
        WorldTensor { ptr, dims: self.dims.clone(), strides: self.strides.clone(), len: self.len, marker: PhantomData}
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

impl<'a, T> WorldSlice<'a, T> {
    pub fn new(tensor: &'a WorldTensor<T>, slices: TsSlices) -> WorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices);
        WorldSlice { world: &tensor, slice }
    }
}

impl<'a, T> MutWorldSlice<'a, T> {
    pub fn new(tensor: &'a mut WorldTensor<T>, slices: TsSlices) -> MutWorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices);
        MutWorldSlice { world: tensor, slice }
    }
}

impl<'a, T> From<*const WorldTensor<T>> for IndexReg<'a> {
    fn from(a: *const WorldTensor<T>) -> Self {
        unsafe {
            let a = &*a;
            IndexReg{
                dims: a.dims.as_ptr(),
                strides: a.strides.as_ptr(),
                len: a.dims.len(),
                nelems: a.len,
                marker: PhantomData}

        }
    }
}

impl<'a, T> From<*const WorldSlice<'a, T>> for IndexSlice<'a> {
    fn from(a: *const WorldSlice<T>) -> Self {
        unsafe {
            let a = &*a;
            IndexSlice { dims: a.slice.sizes.as_ptr(), 
                s_strides: a.slice.strides.as_ptr(), 
                g_strides: a.world.strides.as_ptr(), 
                s_stride_ind: a.slice.ref_inds.as_ptr(),
                s_len: a.slice.strides.len(), 
                linear_offset: a.slice.lin_offset,
                nelems: a.world.len, 
                marker: PhantomData }
        }
    }
}

impl<'a, T> From<*const MutWorldSlice<'a, T>> for IndexSlice<'a> {
    fn from(a: *const MutWorldSlice<T>) -> Self {
        unsafe {
            let a = &*a;
            IndexSlice { dims: a.slice.sizes.as_ptr(), 
                s_strides: a.slice.strides.as_ptr(), 
                g_strides: a.world.strides.as_ptr(),
                s_stride_ind: a.slice.ref_inds.as_ptr(),
                s_len: a.slice.strides.len(), 
                linear_offset: a.slice.lin_offset,
                nelems: a.world.len, 
                marker: PhantomData }
        }
    }
}

impl<'a, T: Tensor<T, &'a usize>> Iterator for TsIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < self.world.nelems() {
            let ptr = self.world.ptr();
            let v = unsafe {&*ptr.add(self.world.lin_index(self.ind))};
            self.ind += 1;
            Some(v)
        } else {
            None
        }
    }
}

impl<'a, T: Tensor<T, &'a usize>> Iterator for MutTsIter<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < self.world.nelems() {
            let ptr = self.world.ptr() as *mut T;
            let v = unsafe {&mut *ptr.add(self.ind)};
            self.ind += 1;
            Some(v)
        } else {
            None
        }
    }
}


impl<'a, T: 'a, I> Index<I> for WorldTensor<T> 
where I: TsIndex,
    <<I as TsIndex>::IndexT as TileIndex>::Meta: From<*const WorldTensor<T>>
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        let i = ind.convert().tile_cartesian((self as *const WorldTensor<T>).into());
        if let Some(idx) = i {
            return inbounds_get_ptr(&self.ptr, idx);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, I> IndexMut<I> for WorldTensor<T> 
where I: TsIndex,
    <<I as TsIndex>::IndexT as TileIndex>::Meta: From<*const WorldTensor<T>>
{
    fn index_mut(&mut self, ind: I) -> &mut T {
        if let Some(idx) = ind.convert().tile_cartesian((self as *const WorldTensor<T>).into()) {
            return inbounds_get_ptr_mut(&mut self.ptr, idx);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T, I> Index<I> for WorldSlice<'a, T> 
where I: TsIndex,
    ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const WorldSlice<'a, T>>
{
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        if let Some(idx) = index.convert().mark_slice().tile_cartesian((self as *const WorldSlice<'a, T>).into()) {
            return inbounds_get_ptr(&self.world.ptr, idx + self.slice.lin_offset);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T, I> Index<I> for MutWorldSlice<'a, T> 
where I: TsIndex,
    ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const MutWorldSlice<'a, T>>
{
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        let t = index.convert().mark_slice().tile_cartesian((self as *const MutWorldSlice<'a, T>).into());
        if let Some(idx) = t{
            return inbounds_get_ptr(&self.world.ptr, idx + self.slice.lin_offset);
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<'a, T, I> IndexMut<I> for MutWorldSlice<'a, T> 
where I: TsIndex,
    ind::SliceMarker<<I as TsIndex>::IndexT>: TileIndex,
    <ind::SliceMarker<<I as TsIndex>::IndexT> as TileIndex>::Meta: From<*const MutWorldSlice<'a, T>>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        if let Some(idx) = index.convert().mark_slice().tile_cartesian((self as *const MutWorldSlice<'a, T>).into()) {
            return inbounds_get_ptr_mut(&mut self.world.ptr, idx + self.slice.lin_offset);
        } else {
            panic!("Index out of bounds");
        }
    }
}

// Have to watchout for the case where the size of the slice is 0, 
// as that state represents the removal of a dimension through slicing
// similar to the case i:i+1, but that case preserves the dimension
fn construct_slice(dims: &[usize], strides: &[usize], slices: TsSlices) -> Slice {
    let mut offsets = Vec::<usize>::new();
    let mut sizes = Vec::<usize>::new();
    let mut refer_inds = Vec::<usize>::new();

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
                refer_inds.push(i);
            },
            IndexBounds(Edge, S(r)) => {
                let e = wrap(*r, d, i);
                offsets.push(0);
                sizes.push(e);
                refer_inds.push(i);
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
                    refer_inds.push(i);
                }
            }
        };
    }
    for j in slices.len()..dims.len() {
        offsets.push(0);
        sizes.push(dims[j]);
    }
    let strides = compute_strides(&sizes);
    Slice{offsets, sizes, non_zero_dims: new_dims, ref_inds: refer_inds, lin_offset, strides}
}

fn construct_slice_from_slice(s_sizes: &[usize], s_offsets: &[usize], s_ref_inds: &[usize], g_strides: &[usize], old_lin_offset:usize, slices: TsSlices) -> Slice {
    let mut offsets = Vec::<usize>::new();
    let mut sizes = Vec::<usize>::new();
    let mut refer_inds = Vec::<usize>::new();

    let mut new_dims = s_sizes.len();
    let mut lin_offset = 0usize;

    let wrap = |x: isize, d, i| {
        if x.abs() as usize > d {
            panic!("pos {} of slice {} is out of bounds", slices[i], slices);
        }
        if x < 0 {d + x as usize}
        else {x as usize}
    };

    for (s, i) in (*slices).iter().zip(0..slices.len()) {
        let d = s_sizes[i];
        match s {
            IndexBounds(Edge, Edge) => {offsets.push(s_offsets[i]); sizes.push(d)},
            IndexBounds(S(l), Edge) => {
                let e = wrap(*l, d, i);
                offsets.push(e + s_offsets[i]);
                sizes.push(d-e);
                refer_inds.push(s_ref_inds[i]);
            },
            IndexBounds(Edge, S(r)) => {
                let e = wrap(*r, d, i);
                offsets.push(s_offsets[i]);
                sizes.push(e);
                refer_inds.push(s_ref_inds[i]);
            },
            IndexBounds(S(l), S(r)) => {
                if l > r {
                    panic!("left index {} is greater than right {} index for slice {}", l, r, slices);
                }
                let e1 = wrap(*l, d, i);

                if l == r {
                    new_dims -= 1;
                    lin_offset += g_strides[s_ref_inds[i]] * e1;
                } else {
                    offsets.push(e1 + s_offsets[i]);
                    sizes.push(wrap(*r, d, i) - e1);
                    refer_inds.push(s_ref_inds[i]);
                }
            }
        };
    }
    for j in slices.len()..s_sizes.len() {
        offsets.push(0);
        sizes.push(s_sizes[j]);
    }
    let strides = compute_strides(&offsets);
    lin_offset += old_lin_offset;
    Slice{offsets, sizes, non_zero_dims: new_dims, ref_inds: refer_inds, lin_offset, strides}
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![];
    strides.reserve_exact(dims.len());
    let mut k = 1usize;
    for i in (0..dims.len()).rev() {
        strides.push(k);
        k *= dims[i];
    }
    strides.reverse();
    strides
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