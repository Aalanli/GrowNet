use std::ops::Deref;
use std::marker::PhantomData;

// This macro automatically makes
// unit polymorphic structs with deref auto derived
macro_rules! auto_deref {
    ($st_nm:ident) => {
        pub struct $st_nm<T>(T);

        impl<T> Deref for $st_nm<T> {
            type Target = T;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
    ($st_nm:ident($($tps:ty),*)) => {
        {
            struct $st_nm(($($tps),*));
            impl Deref for $st_nm {
                type Target = ($($tps),*);
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }
        }
    };
}

// This trait converts all the possible indexing types
// [usize; N], Vec<usize>, &[usize], etc, into only a few
// manageable representations, that does not check bounds when indexed
// (usually these representations are raw pointers)
pub trait TsIndex {
    type IndexT: TileIndex;
    fn convert(&self) -> Self::IndexT;
    fn mark_slice(s: Self::IndexT) -> SliceMarker<Self::IndexT> {
        SliceMarker(s)
    }
}

// TileIndex converts the representations generated by TsIndex
// into actual linear indices, depending on the type marker
// for example, the implementation for SliceMarker<UniversalIndex<T>> is different
// to UniversalIndex<T>
pub trait TileIndex {
    type Meta;
    fn tile_cartesian(&self, meta: Self::Meta) -> Option<usize>;
}

#[derive(Clone, Copy)]
struct UniversalIndex<'a, T> {
    ptr: *const T,
    len: usize,
    marker: PhantomData<&'a T>
}
#[derive(Clone, Copy)]
struct StaticUIndex<'a, T, const N: usize> {
    ptr: *const T,
    marker: PhantomData<&'a T>
}
// type signifying that the index is indexed within a tensor slice
auto_deref!(SliceMarker);

// regular indexing where the memory order is contiguous and linear
pub struct IndexReg<'a> {
    pub dims: *const usize,
    pub strides: *const usize,
    pub len: usize,
    pub nelems: usize,
    pub marker: PhantomData<&'a usize>
}

// indexing on slices where the memory order is not contiguous
pub struct IndexSlice<'a> {
    pub dims: *const usize,
    pub s_strides: *const usize,
    pub g_strides: *const usize,
    pub len: usize,
    pub nelems: usize,
    pub marker: PhantomData<&'a usize>
}

// regular indexing where the memory order is contiguous and linear
// but the size of the indices are static
struct StaticIndexReg<'a, const N: usize> {
    dims: *const usize,
    strides: *const usize,
    marker: PhantomData<&'a usize>
}

// indexing on slices where the memory order is not contiguous
// the sizes of the meta variables are static
struct StaticIndexSlice<'a, const N: usize> {
    dims: *const usize,
    s_strides: *const usize,
    g_strides: *const usize,
    marker: PhantomData<&'a usize>
}

impl<'a> TileIndex for UniversalIndex<'a, usize> {
    type Meta = IndexReg<'a>;
    fn tile_cartesian(&self, meta: IndexReg<'a>) -> Option<usize> {
        unsafe { 
            if meta.len < self.len {
                return None;
            }
            let mut idx: usize = 0;
            let offset = meta.len - self.len;
            for i in 0..self.len {
                let j = i + offset;
                if *self.ptr.add(i) >= *meta.dims.add(j) {
                    return None;
                }
                idx += *meta.strides.add(j) * *self.ptr.add(i);
            }
            Some(idx)
         }
    }
}

impl<'a, const N: usize> TileIndex for StaticUIndex<'a, usize, N> {
    type Meta = IndexReg<'a>;
    fn tile_cartesian(&self, meta: IndexReg<'a>) -> Option<usize> {
        unsafe {
            // N equals 0 represents the case where a single usize is used
            // to index into the tensor, which just represents linear
            // indexing
            if N == 0 {
                let idx = *self.ptr;
                if idx >= meta.nelems {
                    return None;
                }
                return Some(idx);
            } else { // otherwise, an multidimensional index is used
                if meta.len < N {
                    return None;
                }
                let mut idx: usize = 0;
                let offset = meta.len - N;
                for i in 0..N {
                    let j = i + offset;
                    if *self.ptr.add(i) >= *meta.dims.add(j) {
                        return None;
                    }
                    idx += *meta.strides.add(j) * *self.ptr.add(i);
                }
                return Some(idx)
            }
         }
    }
}

impl<'a> TileIndex for SliceMarker<UniversalIndex<'a, usize>> {
    type Meta = IndexSlice<'a>;
    fn tile_cartesian(&self, meta: IndexSlice<'a>) -> Option<usize> {
        if meta.len < self.len {
            return None;
        }
        let mut idx: usize = 0;
        let offset = meta.len - self.len;
        unsafe { 
            for i in 0..self.len {
                let j = i + offset;
                if *self.ptr.add(i) >= *meta.dims.add(j) {
                    return None;
                }
                idx += *meta.g_strides.add(j) * *self.ptr.add(i);
            }
            Some(idx)
         }
    }
}

impl<'a, const N: usize> TileIndex for SliceMarker<StaticUIndex<'a, usize, N>> {
    type Meta = IndexSlice<'a>;
    fn tile_cartesian(&self, meta: IndexSlice<'a>) -> Option<usize> {
        unsafe {
            // if N equals 0, this represents indexing with a usize, which
            // means indexing into a non-continuous block of memory
            // represented using a slice. This has performance penalities
            // and is supposed to be used with iterators with slices,
            // if used at all.
            if N == 0 {
                let mut idx = 0;
                let mut ind = *self.ptr;
                for i in 0..meta.len {
                    let c = ind / *meta.s_strides.add(i) as usize;
                    idx += c * *meta.g_strides.add(i);
                    ind -= c * *meta.s_strides.add(i);
                    if ind <= 0 {
                        break;
                    }
                }
                if idx >= meta.nelems {
                    return None;
                }
                return Some(idx);
            } else { // otherwise more efficient multi-dimensional indexing into slices
                if meta.len < N {
                    return None;
                }
                let mut idx: usize = 0;
                let offset = meta.len - N;
                for i in 0..N {
                    let j = i + offset;
                    if *self.ptr.add(i) >= *meta.dims.add(j) {
                        return None;
                    }
                    idx += *meta.g_strides.add(j) * *self.ptr.add(i);
                }
                return Some(idx);
            }
         }
    }
}

impl<'a> TsIndex for &'a Vec<usize> {
    type IndexT = UniversalIndex<'a, usize>;
    fn convert(&self) -> Self::IndexT {
        UniversalIndex{ptr: self.as_ptr(), len: self.len(), marker: PhantomData}
    }
}

impl<'a> TsIndex for &'a [usize] {
    type IndexT = UniversalIndex<'a, usize>;
    fn convert(&self) -> Self::IndexT {
        UniversalIndex{ptr: self.as_ptr(), len: self.len(), marker: PhantomData}
    }
}

impl<'a, const N: usize> TsIndex for &'a [usize; N] {
    type IndexT = StaticUIndex<'a, usize, N>;
    fn convert(&self) -> Self::IndexT {
        StaticUIndex{ptr: self.as_ptr(), marker: PhantomData}
    }
}

impl<'a> TsIndex for &'a usize {
    type IndexT = StaticUIndex<'a, usize, 0>;
    fn convert(&self) -> Self::IndexT {
        StaticUIndex{ptr: **self as *const usize, marker: PhantomData}
    }
}