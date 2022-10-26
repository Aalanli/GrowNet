use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::ops::{Index, IndexMut};
use std::iter::Iterator;
use std::fmt::Display;

use super::index2::{self as ind, ConvertIndex, Index as TIndex, LinArr};
use super::slice::{IndexBounds, TsSlices, Slicev2, Edge, S};
use super::{Tensor, MutTensor};

/////////////////////////////////////////////////////////////////////
// The Tensor implementation which owns its datatypes, and is represented as a multidimensional array
// Supports multidimensional indexing and slicing
/////////////////////////////////////////////////////////////////////
pub struct WorldTensor<T> {
    pub ptr: NonNull<T>,         // NonNull pointer
    pub dims: Vec<usize>,        // dimensions row-major
    pub strides: Vec<usize>,     // strides row-major
    pub nelems: usize,              // total n elements
    pub marker: PhantomData<T>,  // tells compiler that Tensor owns values of type T
}

impl<T> WorldTensor<T> {
    pub fn new<F: Fn() -> T>(dims: Vec<usize>, elem_fn: F) -> WorldTensor<T> {
        let strides= compute_strides(&dims);
        let nelems = strides[0] * dims[0];
        assert!(nelems > 0, "cannot make 0 sized tensor");

        let layout = Layout::array::<T>(nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };
        let raw_ptr = ptr.as_ptr();
        unsafe {
            for i in 0..nelems {
                *raw_ptr.add(i) = elem_fn();
            }
        }

        WorldTensor{ptr, dims, strides, nelems, marker: PhantomData}
    }

    fn index_params<'t>(&self) -> ind::LinArr<'t> {
        ind::LinArr { dims: self.dims.as_ptr(), strides: self.strides.as_ptr(), len: self.dims.len(), _marker: PhantomData }
    }

    pub fn slice(&self, slices: &TsSlices) -> WorldSlice<T> {
        WorldSlice::new(self, slices)
    }

    pub fn slice_mut(&mut self, slices: &TsSlices) -> MutWorldSlice<T> {
        MutWorldSlice::new(self, slices)
    }

    pub fn iter(&self) -> TsIter<'_, Self> {
        TsIter { world: self, ind: 0, nelems: self.nelems }
    }

    pub fn iter_mut(&mut self) -> MutTsIter<'_, Self> {
        let nelems = self.dims.iter().product();
        MutTsIter { world: self, ind: 0, nelems: nelems }
    }
}

impl<T: Display> Display for WorldTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        recursive_write(f, self.ptr.as_ptr(), 0, 0, &self.dims, &self.strides, self.nelems);
        write!(f, "")
    }
}

impl<T: Clone> Clone for WorldTensor<T> {
    fn clone(&self) -> Self {
        let layout = Layout::array::<T>(self.nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };
        for i in 0..self.nelems {
            unsafe {
                *ptr.as_ptr().add(i) = (*self.ptr.as_ptr().add(i)).clone();
            }
        }
        WorldTensor { ptr, dims: self.dims.clone(), strides: self.strides.clone(), nelems: self.nelems, marker: PhantomData}
    }
}

impl<T> Drop for WorldTensor<T> {
    fn drop(&mut self) {
        for i in 0..self.nelems {
            unsafe{ ptr::drop_in_place(self.ptr.as_ptr().add(i)); }
        }
        let layout = Layout::array::<T>(self.nelems).unwrap();
        unsafe {
            alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

/////////////////////////////////////////////////////////////////////
// Indexing operations for WorldTensor, always checks inbounds and is always valid / safe
/////////////////////////////////////////////////////////////////////
impl<'a, T, I: 'a> Index<I> for WorldTensor<T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<LinArr<'a>>
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        let uindex = ind.convert();
        let index_params = self.index_params();
        let i = TIndex::<LinArr>::tile_cartesian(&uindex, &index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.nelems {
                return inbounds_get_ptr(&self.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}

impl<'a, T, I: 'a> IndexMut<I> for WorldTensor<T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<LinArr<'a>>
{
    fn index_mut(&mut self, ind: I) -> &mut T {
        let uindex = ind.convert();
        let index_params = self.index_params();
        let i = TIndex::<LinArr>::tile_cartesian(&uindex, &index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.nelems {
                return inbounds_get_ptr_mut(&mut self.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}

impl<'a, T, I: ConvertIndex<'a> + 'a> Tensor<T, I> for WorldTensor<T>
where <I as ConvertIndex<'a>>::Result: TIndex<LinArr<'a>>
{
    fn slice(&self, slices: &TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice(&self.dims, &self.strides, slices, None);
        WorldSlice { world: self, slice }
    }
    fn iter(&self) -> TsIter<Self> {
        self.iter()
    }
    fn nelems(&self) -> usize {
        self.nelems
    }
    fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
    unsafe fn inbounds(&self, idx: usize) -> &T {
        &*self.ptr.as_ptr().add(idx)
    }
}

impl<'a, T, I: ConvertIndex<'a> + 'a> MutTensor<T, I> for WorldTensor<T> 
where <I as ConvertIndex<'a>>::Result: TIndex<LinArr<'a>>
{
    fn iter_mut(&mut self) -> MutTsIter<Self> {
        self.iter_mut()
    }
    fn slice_mut(&mut self, slices: &TsSlices) -> MutWorldSlice<'_, T> {
        let slice = construct_slice(&self.dims, &self.strides, slices, None);
        MutWorldSlice { world: self, slice }
    }
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
    unsafe fn inbounds_mut(&mut self, idx: usize) -> &mut T {
        &mut *self.ptr.as_ptr().add(idx)
    }
}

/////////////////////////////////////////////////////////////////////
// WorldSlice represents a view of a WorldTensor, the memory order may not be contiguous
/////////////////////////////////////////////////////////////////////
pub struct WorldSlice<'a, T> {
    world: &'a WorldTensor<T>,
    pub slice: Slicev2
}

pub struct MutWorldSlice<'a, T> {
    world: &'a mut WorldTensor<T>,
    pub slice: Slicev2
}

impl<'a, T> WorldSlice<'a, T> {
    pub fn new(tensor: &'a WorldTensor<T>, slices: &TsSlices) -> WorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices, None);
        WorldSlice { world: tensor, slice }
    }

    fn index_params<'t>(&self) -> ind::Slice<'t> {
        ind::Slice { 
            s_dims: self.slice.sizes.as_ptr(),
            s_offsets: self.slice.offsets.as_ptr(),
            g_strides: self.slice.g_strides.as_ptr(),
            s_len: self.slice.sizes.len(),
            lin_offsets: self.slice.lin_offset,
            _marker: PhantomData
        }
    }

    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter { slice: &self, cur_dim: (0..self.slice.g_strides.len()).map(|_| 0).collect() }
    }

    pub fn slice(&self, slices: &TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice, slices);
        WorldSlice { world: self.world, slice }
    }
}

impl<'a, T> MutWorldSlice<'a, T> {
    pub fn new(tensor: &'a mut WorldTensor<T>, slices: &TsSlices) -> MutWorldSlice<'a, T> {
        let slice = construct_slice(&tensor.dims, &tensor.strides, slices, None);
        MutWorldSlice { world: tensor, slice }
    }

    fn index_params<'t>(&self) -> ind::Slice<'t> {
        ind::Slice { 
            s_dims: self.slice.sizes.as_ptr(),
            s_offsets: self.slice.offsets.as_ptr(),
            g_strides: self.slice.g_strides.as_ptr(),
            s_len: self.slice.sizes.len(),
            lin_offsets: self.slice.lin_offset,
            _marker: PhantomData
        }
    }

    pub fn immutable(&self) -> WorldSlice<'_, T> {
        WorldSlice { world: self.world, slice: self.slice }
    }

    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter { slice: &self.immutable(), cur_dim: (0..self.slice.g_strides.len()).map(|_| 0).collect() }
    }

    pub fn iter_mut(&mut self) -> MutTsIter<'_, T> {
        MutSliceIter { slice: &mut self, cur_dim: (0..self.slice.g_strides.len()).map(|_| 0).collect() }
    }

    pub fn slice(&self, slices: &TsSlices) -> WorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice, slices);
            WorldSlice{ world: self.world, slice }
    }

    pub fn slice_mut(&mut self, slices: &TsSlices) -> MutWorldSlice<'_, T> {
        let slice = construct_slice_from_slice(
            &self.slice, slices);
            MutWorldSlice{ world: self.world, slice }
    }
}

// Indexing for Worldslice and MutWorldSlice
// where the array param type is a slice
impl<'a, T, I> Index<I> for WorldSlice<'a, T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        let uindex = ind.convert();
        let index_params = self.index_params();
        let i = TIndex::<ind::Slice>::tile_cartesian(&uindex, &index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.world.nelems {
                return inbounds_get_ptr(&self.world.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}

// the index function is exactly the same as the previous block
impl<'a, T, I> Index<I> for MutWorldSlice<'a, T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        let uindex = ind.convert();
        let index_params = self.index_params();
        let i = TIndex::<ind::Slice>::tile_cartesian(&uindex, &index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.world.nelems {
                return inbounds_get_ptr(&self.world.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}

// this is also very similar to the previous block, copy & paste saves the day
impl<'a, T, I> IndexMut<I> for MutWorldSlice<'a, T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    fn index_mut(&mut self, ind: I) -> &mut T {
        let uindex = ind.convert();
        let index_params = self.index_params();
        let i = TIndex::<ind::Slice>::tile_cartesian(&uindex, &index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.world.nelems {
                return inbounds_get_ptr_mut(&mut self.world.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}

impl<'a, T, I: ConvertIndex<'a>> Tensor<T, I> for WorldSlice<'a, T>
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    fn slice(&self, slices: &TsSlices) -> WorldSlice<'_, T> {
        self.slice(&slices)
    }
    fn iter(&self) -> SliceIter<'_, T> {
        self.iter()
    }
    fn nelems(&self) -> usize {
        self.world.nelems
    }
    fn ptr(&self) -> *const T {
        self.world.ptr.as_ptr()
    }
    unsafe fn inbounds(&self, _idx: usize) -> &T {
        panic!("inbounds for WorldSlice is not implementated")
    }
}

impl<'a, T, I: ConvertIndex<'a> + 'a> Tensor<T, I> for MutWorldSlice<'a, T>
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    fn slice(&self, slices: &TsSlices) -> WorldSlice<'_, T> {
        self.slice(&slices)
    }
    fn iter(&self) -> TsIter<Self> {
        self.iter()
    }
    fn nelems(&self) -> usize {
        self.world.nelems
    }
    fn ptr(&self) -> *const T {
        self.world.ptr.as_ptr()
    }
    unsafe fn inbounds(&self, _idx: usize) -> &T {
        panic!("inbounds for MutWorldSlice is not implementated")
    }
}

impl<'a, T, I: ConvertIndex<'a> + 'a> MutTensor<T, I> for MutWorldSlice<'a, T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: TIndex<ind::Slice<'a>>
{
    fn iter_mut(&mut self) -> MutTsIter<Self> {
        self.iter_mut()
    }
    fn slice_mut(&mut self, slices: &TsSlices) -> MutWorldSlice<'_, T> {
        self.slice_mut(slices)
    }
    fn ptr_mut(&mut self) -> *mut T {
        self.world.ptr.as_ptr()
    }
    unsafe fn inbounds_mut(&mut self, _idx: usize) -> &mut T {
        panic!("inbounds_mut for MutWorldSlice is not implementated")
    }
}

/////////////////////////////////////////////////////////////////////
// Iterator implementations for WorldSlice and WorldTensor
/////////////////////////////////////////////////////////////////////
pub struct TsIter<'a, T> {
    world: &'a T,
    ind: usize,
    nelems: usize,
}

pub struct MutTsIter<'a, T> {
    world: &'a mut T,
    ind: usize,
    nelems: usize,
}

impl<'a, U: 'a, T: Index<usize, Output = U>> Iterator for TsIter<'a, T> {
    type Item = &'a U;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind >= self.nelems {
            return None;
        }
        let t = &self.world[self.ind];
        self.ind += 1;
        Some(t)
    }
}

impl<'a, T> Iterator for MutTsIter<'a, WorldTensor<T>> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.ind >= self.nelems {
            return None;
        }
        let t = unsafe {
            &mut *self.world.ptr.as_ptr().add(self.ind)
        };
        self.ind += 1;
        Some(t)
    }
}

struct SliceIter<'a, T> {
    slice: &'a WorldSlice<'a, T>,
    cur_dim: Vec<usize>,
}

struct MutSliceIter<'a, T> {
    slice: &'a mut MutWorldSlice<'a, T>,
    cur_dim: Vec<usize>
}

impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.slice.slice.g_strides.iter()
                        .zip(self.cur_dim.iter()).fold(0, |acc, (stride, idx)| acc + stride * idx );

                        
        
        // if the first dimension overflows, there are no more elements
        if carry_over(&mut self.cur_dim, &self.slice.slice.sizes) {
            return None;
        }

        let item = unsafe {
            <WorldTensor<T> as Tensor<T, usize>>::inbounds(&self.slice.world, idx)
        };
        Some(item)
    }
}

impl<'a, T> Iterator for MutSliceIter<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.slice.slice.g_strides.iter()
                        .zip(self.cur_dim.iter()).fold(0, |acc, (stride, idx)| acc + stride * idx );

                        
        
        // if the first dimension overflows, there are no more elements
        if carry_over(&mut self.cur_dim, &self.slice.slice.sizes) {
            return None;
        }

        let world = &*self.slice.world as *const WorldTensor<T> as *mut WorldTensor<T>;

        let item = unsafe {
            <WorldTensor<T> as MutTensor<T, usize>>::inbounds_mut(&mut *world, idx)
        };
        Some(item)
    }
}


/////////////////////////////////////////////////////////////////////
/// Helper functions
/////////////////////////////////////////////////////////////////////

/// Increase the index by 1, and carry over to higher dimensions if appliciable
/// returns true if index is full, false otherwise
fn carry_over(index: &mut [usize], dims: &[usize]) -> bool {
    let mut carry_over = true;
    index.iter_mut().rev()
        .zip(dims.iter().rev())
        .for_each(|(x, s)| {
            if carry_over {
                *x += 1;
                carry_over = false;
            }
            if *x >= *s {
                *x = 0;
                carry_over = true;
            }
        });
    carry_over
}


/// Recursive Write uses recursion to print out tensors
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

/// Construct slice produces slicing parameters from tensor parameters
/// Have to watchout for the case where the size of the slice is 0, 
/// as that state represents the removal of a dimension through slicing
/// similar to the case i:i+1, but that case preserves the dimension
fn construct_slice(dims: &[usize], strides: &[usize], slices: &TsSlices, old_offsets: Option<&[usize]>) -> Slicev2 {
    // the offset of each dimension, as a slice may not being on 0, ex [2:], the offset would be 2 + i
    // if the index into that dimension is i
    let mut offsets = Vec::<usize>::new();
    let mut sizes = Vec::<usize>::new();
    // the global stride of that dimension, this is kept since slicing with an integer
    // ex [0], will remove that dimension
    let mut g_strides = Vec::<usize>::new();

    // keep track of the number of dimensions remaining after slicing
    let mut new_dims = dims.len();
    // the linear offset for the dimensions removed, ex. [:, 1, :], any index into 
    // this new 2D slice will have to offset by 1 * strides[i], where strides[i],
    // is the global stride of the contiguous array backing 
    let mut lin_offset = 0;

    // we accept negative indices, so we perform the wrap around operation here
    let wrap = |x: isize, d, i| {
        if x.abs() as usize > d {
            panic!("pos {} of slice {} is out of bounds", slices[i], slices);
        }
        if x < 0 {d + x as usize}
        else {x as usize}
    };

    // if there are any old offsets, we want the new offsets to be added onto the old offsets
    // this is useful when we want to construct a new slice out of an old slice
    let offset = |x: usize, i: usize| {
        match old_offsets {
            Some(xs) => x + xs[i],
            None => x
        }
    };

    for (s, i) in (*slices).iter().zip(0..slices.len()) {
        let d = dims[i];
        match s {
            IndexBounds(Edge, Edge) => {
                offsets.push(offset(0, i)); sizes.push(d); g_strides.push(strides[i]);},
            IndexBounds(S(l), Edge) => {
                let e = wrap(*l, d, i);
                offsets.push(offset(e, i));
                sizes.push(d-e);
                g_strides.push(strides[i]);
            },
            IndexBounds(Edge, S(r)) => {
                let e = wrap(*r, d, i);
                offsets.push(offset(0, i));
                sizes.push(e);
                g_strides.push(strides[i]);
            },
            IndexBounds(S(l), S(r)) => {
                if l > r {
                    panic!("left index {} is greater than right {} index for slice {}", l, r, slices);
                }
                let e1 = wrap(*l, d, i);
                // this is the case where slicing is done with an integer, instead of a range
                if l == r {
                    new_dims -= 1;
                    lin_offset += strides[i] * e1;
                } else {
                    offsets.push(offset(e1, i));
                    sizes.push(wrap(*r, d, i) - e1);
                    g_strides.push(strides[i]);
                }
            }
        };
    }
    // since slicing starts left to right, any dimensions unsliced to the right
    // is viewed as [:]
    for j in slices.len()..dims.len() {
        offsets.push(0);
        sizes.push(dims[j]);
        g_strides.push(strides[j]);
    }
    Slicev2{offsets, sizes, non_zero_dims: new_dims, g_strides, lin_offset}
}

/// For cases where it may be helpful to slice a slice
fn construct_slice_from_slice(old_slice: &Slicev2, slices: &TsSlices) -> Slicev2 {
    let mut new_slice = construct_slice(
        &old_slice.sizes, &old_slice.g_strides, slices, Some(&old_slice.offsets));
    new_slice.lin_offset += old_slice.lin_offset;
    new_slice
}

/// Computes the stride from dimensions, the last stride is always 1
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

/// This function does not check if the index is inbounds
fn inbounds_get_ptr<T>(ptr: &NonNull<T>, i: usize) -> &T {
    unsafe{ ptr.as_ptr().add(i).as_ref().unwrap() }
}

/// This function does not check if the index is inbounds
fn inbounds_get_ptr_mut<T>(ptr: &mut NonNull<T>, i: usize) -> &mut T {
    unsafe { ptr.as_ptr().add(i).as_mut().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::slice::slice;
    #[test]
    fn slice_test() {
        let mut ts = WorldTensor::new(vec![5, 6, 7], || 1.0f32);
        let slice = ts.slice_mut(&slice![.., .., ..]);
        slice.iter_mut().
    }
}