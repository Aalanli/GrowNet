pub mod tensor;
pub mod index;
pub mod slice;

pub(crate) use slice::slice as tslice;

pub use tensor::{WorldTensor, WorldSlice, MutWorldSlice, TsIter, MutTsIter};
pub use index::{TIndex, UnsafeIndex};