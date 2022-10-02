
pub trait ConvertIndex {
    type Result;
    fn convert(&self) -> Self::Result;
}

// Possible index types / representations
#[derive(Clone, Copy)]
pub struct DynIndex {
    ptr: *const usize,
    len: usize,
}
#[derive(Clone, Copy)]
pub struct StatIndex<const N: usize> {
    ptr: *const usize,
}
pub struct LinIndex(usize);

impl ConvertIndex for Vec<usize> {
    type Result = DynIndex;
    fn convert(&self) -> Self::Result {
        DynIndex{ptr: self.as_ptr(), len: self.len()}
    }
}

impl ConvertIndex for [usize] {
    type Result = DynIndex;
    fn convert(&self) -> Self::Result {
        DynIndex{ptr: self.as_ptr(), len: self.len()}
    }
}

impl<const N: usize> ConvertIndex for [usize; N] {
    type Result = StatIndex<N>;
    fn convert(&self) -> Self::Result {
        StatIndex{ptr: self.as_ptr()}
    }
}

impl ConvertIndex for usize {
    type Result = LinIndex;
    fn convert(&self) -> Self::Result {
        LinIndex(*self)
    }
}

// Possible array types / representations
pub struct LinArr{
    pub dims: *const usize,    // dynamic dimensions
    pub strides: *const usize, // dynamic strides
    pub len: usize,            // length of the dims and strides array
}

pub struct Slice{
    pub s_dims: *const usize,    // the dimensions of the slice
    pub s_offsets: *const usize, // the offsets of the slice into global memory
    pub g_strides: *const usize, // the actual stride of the dimenions into global memory
    pub s_len: usize,            // length of the s_dims, s_offsets, and g_strides array
    pub lin_offsets: usize,      // Linearly offset, for any dimensions that may be removed during slicing
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
impl Index<Slice> for DynIndex {
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
        // if there are more indicies than dimensions in the slice
        // then this will under-flow, causing an opaque error
        // maybe check for it in advance?
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
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// This is just like the dynamic case, except for a static number
// of index dimensions, which will hopefully result in loop unrolling
impl<const N: usize> Index<Slice> for StatIndex<N> {
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
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
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// This case is if we index linearly into the slice, as if the memory
// order was contiguous, which it is not. Which actually results
// in similar or worse performance than the multidimensional index case
// since we have to wrap around each dimension.
impl Index<Slice> for LinIndex {
    fn tile_cartesian(&self, arr: &Slice) -> Option<usize> {
        let mut ind = self.0;
        let mut s_stride = 1;
        // the index into contiguous memory
        let mut idx = 0;
        unsafe {
            for i in (1..arr.s_len).rev() {
                let dim_i = *arr.s_dims.add(i);
                // this the slicing index in dimension i
                let si_idx = ind % (dim_i * s_stride);
                ind -= si_idx * s_stride;
                idx += (si_idx + *arr.s_offsets.add(i)) * *arr.g_strides.add(i);
                s_stride *= dim_i;
            }
            let last_ind = ind / s_stride;
            if last_ind > *arr.s_dims {
                return None;
            }
            idx += (last_ind + *arr.s_offsets) * *arr.g_strides;
        }
        idx += arr.lin_offsets;
        Some(idx)
    }
}

// Indexing into contiguous memory now, much easier than slice views
impl Index<LinArr> for DynIndex {
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
impl<const N: usize> Index<LinArr> for StatIndex<N> {
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

// the simplist case, linear indexing into linear, contiguous memory
// however, all the other ones produce inbounds indices, by nature
// of their construction, but this one doesn't
impl Index<LinArr> for LinIndex {
    fn tile_cartesian(&self, _: &LinArr) -> Option<usize> {
        Some(self.0)
    }
}