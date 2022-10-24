pub mod slice;
pub mod index2;
pub mod tensor2;

pub(crate) use slice::slice as tslice;

pub use tensor2::{WorldTensor, WorldSlice, MutWorldSlice, TsIter, MutTsIter};
pub use tensor2 as tensor;
pub use index2 as index;
pub use slice::Slicev2 as Slice;

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