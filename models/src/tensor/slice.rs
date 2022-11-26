use std::fmt::Display;
use std::ops::{Deref, Range, RangeFrom, RangeTo, RangeFull};

// IndexInfo represents the raw slices which the user would input
// Since the user usually inputs relative position slices in the 
// form of [i:j, w:, k:, :, ...], S(i) the cases where a slice
// is present, and Edge represents the cases where it is not
pub enum IndexInfo {
    Edge,
    S(isize)
}

pub use IndexInfo::Edge as Edge;
pub use IndexInfo::S as S;

// Similarly IndexBounds represents the pair of relative slices
// there are five cases i:j, i:, :j, :, i
// where i is represented as IndexBounds(S(i),S(i)), for taking
// the slices of a whole dimension. 
pub struct IndexBounds(pub IndexInfo, pub IndexInfo);


// A static arrow of pairs representing a slice of an entire tensor
// The default is static as the TsSlices generated by the macro, 
// which is the most common way of generating a slice, is static,
// so performance optimizations can be used
pub struct TsSlices(pub Vec<IndexBounds>);
// A vector of pairs representing the slice of an entire tensor

// struct representing absolute indicies to the sliced tensor
// in contrast to TsSlices which represents relative slices
#[derive(Debug)]
pub struct Slice {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub ref_inds: Vec<usize>,
    pub non_zero_dims: usize,
    pub lin_offset: usize,
}

#[derive(Debug, Clone)]
pub struct Slicev2 {
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub g_strides: Vec<usize>,
    pub non_zero_dims: usize,
    pub lin_offset: usize
}

impl Deref for TsSlices {
    type Target = Vec<IndexBounds>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for IndexBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let S(i) = self.0 {
            if let S(j) = self.1 {
                if i == j {
                    return write!(f, "{}", i);
                }
                return write!(f, "{}:{}", i, j);
            }
            return write!(f, "{}:", i);
        }
        let r = if let S(i) = self.1 {format!("{}", i)} else {String::from("")};
        write!(f, ":{}", r)
    }
    
}

impl Display for TsSlices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[").unwrap();
        for i in 0..self.len()-1 {
            write!(f, "{}, ", self[i]).unwrap();
        }
        write!(f, "{}]", self[self.len()-1])
    }
}

// SliceSpec converts different range types to a unifying IndexBounds type
// For each of the five cases listed above IndexBounds.
pub trait SliceSpec {
    type Output;
    fn to_slice(&self) -> Self::Output;
}

impl SliceSpec for Range<isize> {
    type Output = IndexBounds;
    fn to_slice(&self) -> Self::Output {
        IndexBounds(S(self.start), S(self.end))
    }
}

impl SliceSpec for RangeTo<isize> {
    type Output = IndexBounds;
    fn to_slice(&self) -> Self::Output {
        IndexBounds(Edge, S(self.end))
    }
}

impl SliceSpec for RangeFrom<isize> {
    type Output = IndexBounds;
    fn to_slice(&self) -> Self::Output {
        IndexBounds(S(self.start), Edge)
    }
}

impl SliceSpec for RangeFull {
    type Output = IndexBounds;
    fn to_slice(&self) -> Self::Output {
        IndexBounds(Edge, Edge)
    }
}

impl SliceSpec for isize {
    type Output = IndexBounds;
    fn to_slice(&self) -> Self::Output {
        IndexBounds(S(*self), S(*self))
    }
}

// slice![1..2, 3..4, 2..3], etc. generates a TsSlices struct
macro_rules! slice {
    ($($r:expr),*) => {
        {
            crate::tensor::slice::TsSlices(vec![$(crate::tensor::slice::SliceSpec::to_slice(&($r))),*])
        }
    };
}

pub(crate) use slice;
