/// All the logic associated with Tensors, mainly indexing, slicing and iterators
pub mod tensor;
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
mod index;

/// The macro which converts easy syntax, *eg.* slice![0..2, -2..-1]
/// to the internal representation
pub(crate) use slice::slice as slice;

pub use tensor::{WorldTensor, WorldSlice, MutWorldSlice, TsIter, MutTsIter};
use slice::Slicev2 as Slice;

use std::ops::{Index, IndexMut};