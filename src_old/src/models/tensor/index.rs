/// Indexing is implemented using genrics to reduce boilerplate (perhaps unsuccessfully) 
/// as the ConvertIndex trait converts disparate collections into an unified representation
/// of which there are three possibilities, a dynamically sized multidimensional index
/// a statically sized multidimensional index, and a linear index type. 
/// Dynamically sized (DynIndex), and statically sized (StatIndex) indicate multidimensional
/// while the LinIndex type represent linear indexing into contiguous memory.
/// To extend indexing for another type, simply implement the ConvertIndex trait, as the Index
/// trait takes care of computing the indicies, whether cartesian or linear.

/// Right now, the structure is similar to this
/// 
/// Index representations | Unified Index reprs | Array Representations 
/// Vec<usize>     -----> | DynIndex     -----> | (LinArr, Slice)
/// &[usize]       -----> | DynIndex     -----> | (LinArr, Slice)
/// &[usize; N]    -----> | StatIndex<N> -----> | (LinArr, Slice)
/// usize          -----> | LinIndex     -----> | (LinArr, Slice)
/// 
/// Since rightnow there are only two tensor representations, a linear, contiguous memory layout
/// and a view of contiguous memory
use std::marker::PhantomData;

pub trait ConvertIndex {
    type Result<'a>;
    fn convert<'a>(&self) -> Self::Result<'a>;
}
/////////////////////////////////////////////////
/// Possible index types / representations
/////////////////////////////////////////////////

#[derive(Clone, Copy)]
pub struct DynIndex<'a> {
    ptr: *const usize,
    len: usize,
    _marker: PhantomData<&'a usize>
}
#[derive(Clone, Copy)]
pub struct StatIndex<'a, const N: usize> {
    ptr: *const usize,
    _marker: PhantomData<&'a usize>

}
pub struct LinIndex<'a>(usize, PhantomData<&'a usize>);

impl ConvertIndex for Vec<usize> {
    type Result<'a> = DynIndex<'a>;
    fn convert<'a>(&self) -> Self::Result<'a> {
        DynIndex{ptr: self.as_ptr(), len: self.len(), _marker: PhantomData}
    }
}

impl ConvertIndex for [usize] {
    type Result<'a> = DynIndex<'a>;
    fn convert<'a>(&self) -> Self::Result<'a> {
        DynIndex{ptr: self.as_ptr(), len: self.len(), _marker: PhantomData}
    }
}

impl<const N: usize> ConvertIndex for [usize; N] {
    type Result<'a> = StatIndex<'a, N>;
    fn convert<'a>(&self) -> Self::Result<'a> {
        StatIndex{ptr: self.as_ptr(), _marker: PhantomData}
    }
}

impl ConvertIndex for usize {
    type Result<'a> = LinIndex<'a>;
    fn convert<'a>(&self) -> Self::Result<'a> {
        LinIndex(*self, PhantomData)
    }
}

/////////////////////////////////////////////////
/// Possible array types / representations
/////////////////////////////////////////////////


pub struct LinArr<'a> {
    pub dims: *const usize,    // dynamic dimensions
    pub strides: *const usize, // dynamic strides
    pub len: usize,            // length of the dims and strides array
    pub _marker: PhantomData<&'a usize>
}

// A tensor with a static number of dimensions
pub struct StatArr<'a, const D: usize> {
    pub dims: *const usize,
    pub strides: *const usize,
    pub _marker: &'a usize,
}

pub struct Slice<'a> {
    pub s_dims: *const usize,    // the dimensions of the slice
    pub s_offsets: *const usize, // the offsets of the slice into global memory
    pub g_strides: *const usize, // the actual stride of the dimenions into global memory
    pub s_len: usize,            // length of the s_dims, s_offsets, and g_strides array
    pub lin_offsets: usize,      // Linearly offset, for any dimensions that may be removed during slicing
    pub _marker: PhantomData<&'a usize>
}

// the indexing trait, where self represents the index representation
// and Arr represents the array representation
pub trait Index<Arr> {
    fn tile_cartesian(&self, arr: &Arr) -> Option<usize>;
}

// If the length of the index is smaller than the number of dimensions
// in the slice, then we compute the index as if we padded the
// index array to the left with zeros, to match the number of slicing
// dimensions
impl<'a> Index<Slice<'a>> for DynIndex<'a> {
    #[inline]
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
        // yes, there is a case where s_len is equal to 0, need to check for
        // that to avoid underflow
        if arr.s_len < self.len {
            return None;
        }
        let offset = arr.s_len - self.len;
        let mut idx = 0;
        for i in 0..self.len {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.s_dims.add(j) {
                    return None;
                }
                let stride = *arr.g_strides.add(j);
                let offset = *arr.s_offsets.add(j);
                idx += (ind_i + offset) * stride;
            }
        }
        // still need increase the index by any potential offsets
        for i in 0..offset {
            unsafe {
                let stride = *arr.g_strides.add(i);
                let offset = *arr.s_offsets.add(i);
                idx += offset * stride;
            }
        }
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// This is just like the dynamic case, except for a static number
// of index dimensions, which will hopefully result in loop unrolling
impl<'a, const N: usize> Index<Slice<'a>> for StatIndex<'a, N> {
    #[inline]
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
        // yes, there is a case where s_len is equal to 0, need to check for
        // that to avoid underflow
        if arr.s_len < N {
            return None;
        }
        let offset = arr.s_len - N;
        let mut idx = 0;
        for i in 0..N {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.s_dims.add(j) {
                    return None;
                }
                let stride = *arr.g_strides.add(j);
                let offset = *arr.s_offsets.add(j);
                idx += (ind_i + offset) * stride;
            }
        }
        // still need increase the index by any potential offsets
        for i in 0..offset {
            unsafe {
                let stride = *arr.g_strides.add(i);
                let offset = *arr.s_offsets.add(i);
                idx += offset * stride;
            }
        }
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// This case is if we index linearly into the slice, as if the memory
// order was contiguous, which it is not. Which actually results
// in similar or worse performance than the multidimensional index case
// since we have to wrap around each dimension.
impl<'a> Index<Slice<'a>> for LinIndex<'a> {
    #[inline]
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
        let mut ind = self.0;
        // the index into contiguous memory
        let mut idx = 0;
        // alas if s_len is 0, thent this crashes, since the arrays would have a size of 0 
        // as well
        if arr.s_len == 0 && self.0 == 0 {
            return Some(arr.lin_offsets);
        } else if self.0 > 0 {
            return None;
        }
        unsafe {
            for i in (1..arr.s_len).rev() {
                let last_dim = *arr.s_dims.add(i);
                // this the slicing index in dimension i
                let over = ind % last_dim;
                ind -= over;
                ind /= last_dim;
                idx += (over + *arr.s_offsets.add(i)) * *arr.g_strides.add(i);
            }
            let last_dim = *arr.s_dims;
            if ind >= last_dim {
                return None;
            }
            idx += (ind + *arr.s_offsets) * *arr.g_strides;
        }
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// Indexing into contiguous memory now, much easier than slice views
impl<'a> Index<LinArr<'a>> for DynIndex<'a> {
    #[inline]
    fn tile_cartesian(&self, arr: &LinArr) -> Option<usize> {
        let offset = arr.len - self.len;
        let mut idx = 0;
        for i in 0..self.len {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.dims.add(j) {
                    return None;
                }
                let stride = *arr.strides.add(j);
                idx += ind_i * stride;
            }
        }
        Some(idx)
    }
}

// More (hopefully) loop unrolling, yay!
impl<'a, const N: usize> Index<LinArr<'a>> for StatIndex<'a, N> {
    #[inline]
    fn tile_cartesian(&self, arr: &LinArr) -> Option<usize> {
        let offset = arr.len - N;
        let mut idx = 0;
        for i in 0..N {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.dims.add(j) {
                    return None;
                }
                let stride = *arr.strides.add(j);
                idx += ind_i * stride;
            }
        }
        Some(idx)
    }
}

// exactly the same code as before
impl<'a, const D: usize> Index<StatArr<'a, D>> for DynIndex<'a> {
    #[inline]
    fn tile_cartesian(&self, arr: &StatArr<D>) -> Option<usize> {
        let offset = D - self.len;
        let mut idx = 0;
        for i in 0..self.len {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.dims.add(j) {
                    return None;
                }
                let stride = *arr.strides.add(j);
                idx += ind_i * stride;
            }
        }
        Some(idx)
    }
}
// just copy and paste
impl<'a, const N: usize, const D: usize> Index<StatArr<'a, D>> for StatIndex<'a, N> {
    #[inline]
    fn tile_cartesian(&self, arr: &StatArr<D>) -> Option<usize> {
        let offset = D - N;
        let mut idx = 0;
        for i in 0..N {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.dims.add(j) {
                    return None;
                }
                let stride = *arr.strides.add(j);
                idx += ind_i * stride;
            }
        }
        Some(idx)
    }
}


// the simplist case, linear indexing into linear, contiguous memory
// however, all the other ones produce inbounds indices, by nature
// of their construction, but this one doesn't
impl<'a> Index<LinArr<'a>> for LinIndex<'a> {
    fn tile_cartesian(&self, _: &LinArr) -> Option<usize> {
        Some(self.0)
    }
}

impl<'a, const D: usize> Index<StatArr<'a, D>> for LinIndex<'a> {
    fn tile_cartesian(&self, _: &StatArr<D>) -> Option<usize> {
        Some(self.0)
    }
}