pub mod tensor;
pub mod index;
pub mod slice;
pub mod index2;
pub mod tensor2;

pub(crate) use slice::slice as tslice;

pub use tensor2::{WorldTensor, WorldSlice, MutWorldSlice, TsIter, MutTsIter};
