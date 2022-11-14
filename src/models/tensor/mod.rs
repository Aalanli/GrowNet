/// All the logic associated with Tensors, mainly indexing, slicing and iterators
pub mod tensor2;
/// Internals regarding the slice of a tensor, namely the struct that defines a slice
mod slice;
/// The boiler-plate to satisfy the type system for all the common indexing types
/// - Vec<usize>, &Vec<usize>
/// - &[usize]
/// - &[usize; N]
/// - usize
/// 
/// Defines common indexing types for each possibility, namely
/// 1. for linear indexing with usize
/// 2. cartesian indexing with a static array of size N
/// 3. cartesian indexing with a dynamically sized array
mod index2;

mod tensorctx;

/// The macro which converts easy syntax, *eg.* slice![0..2, -2..-1]
/// to the internal representation
pub(crate) use slice::slice as slice;

pub use tensor2::{WorldTensor, WorldSlice, MutWorldSlice, TsIter, MutTsIter};
pub use tensor2 as tensor;
use slice::Slicev2 as Slice;

use std::ops::{Index, IndexMut};


/// The tensor trait aims to support any generic tensor operations one may want
trait Tensor<T, I>: Index<I, Output = T> + Sized {
    fn slice(&self, slices: &slice::TsSlices) -> WorldSlice<'_, T>;
    fn iter(&self) -> TsIter<Self>;
    fn nelems(&self) -> usize;
    fn ptr(&self) -> *const T;
    unsafe fn inbounds(&self, idx: usize) -> &T;
}

trait MutTensor<T, I>: IndexMut<I, Output = T> + Tensor<T, I> {
    fn slice_mut(&mut self, slices: &slice::TsSlices) -> MutWorldSlice<'_, T>;
    fn ptr_mut(&mut self) -> *mut T;
    fn iter_mut(&mut self) -> MutTsIter<Self>;
    unsafe fn inbounds_mut(&mut self, idx: usize) -> &mut T;
}