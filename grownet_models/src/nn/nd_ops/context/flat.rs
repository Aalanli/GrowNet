use super::*;

/// Simple Ctx with static memory allocation, panics if internal buffer size
/// is exceeded
pub struct FlatCtx<T> {
    buf: Vec<T>,
    gen: usize,
}

impl<T: Float> FlatCtx<T> {
    pub fn new(cap: usize) -> Self {
        let mut buf = Vec::new();
        buf.reserve_exact(cap);
        FlatCtx { buf, gen: 0 }
    }

    unsafe fn reserve(&self, nelem: usize) -> usize {
        let cap = self.buf.capacity();
        let len = self.buf.len();
        if len + nelem > cap {
            panic!("not enough memory")
        }
        (&mut *(&self.buf as *const Vec<T> as *mut Vec<T>)).set_len(len + nelem);
        len
    }
    
    /// panics if there is not enough space
    pub fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_shape();
        let raw_dim = dim.raw_dim();
        let nelem = dim.size();
        let idx = unsafe { self.reserve(nelem) };
        let xs = unsafe { self.slice_mut(idx, raw_dim) };
        ArrayViewMut::from_shape(dim, xs).unwrap()
    }

    pub fn zeros<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let mut arr = self.empty(dim);
        arr.fill(T::zero());
        arr
    }

    pub fn clone<'a, D: Dimension>(&'a self, xs: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
        let mut empty = self.empty(xs.raw_dim());
        empty.assign(&xs);
        empty
    }

    // xs is a mutable view because otherwise we would alias memory, if xs was ArrayView for example, it would be possible
    // to create a mutable and immutable view pointing to the same data, since ArrId holds no lifetimes
    pub fn id<D: Dimension>(&self, xs: ArrayViewMut<T, D>) -> ArrId<T, D> {
        let offset = unsafe { 
            let ptr = xs.as_ptr();
            let buf = self.buf.as_ptr();
            ptr.offset_from(buf) 
        };
        if offset < 0 || offset as usize > self.buf.len() {
            // or alternatively, copy xs into self and return the address of that
            panic!("view is out of bounds from current buffer")
        } else {
            ArrId { dim: xs.raw_dim(), offset: offset as usize, gen: self.gen, _data: PhantomData }
        }
    }

    pub fn from_id<D: Dimension>(&self, id: &ArrId<T, D>) -> ArrayView<T, D> {
        assert!(id.gen == self.gen, "generation mismatch between id and ctx");
        let arr = unsafe {
            let ptr = self.buf.as_ptr().add(id.offset);
            ArrayView::from_shape_ptr(id.dim.clone(), ptr)
        };
        arr
    }

    pub fn from_id_mut<D: Dimension>(&self, id: ArrId<T, D>) -> ArrayViewMut<T, D> {
        assert!(id.gen == self.gen, "generation mismatch between id and ctx");
        let arr = unsafe {
            let ptr = self.buf.as_ptr().add(id.offset) as *mut T;
            ArrayViewMut::from_shape_ptr(id.dim.clone(), ptr)
        };
        arr
    }

    unsafe fn slice<D: Dimension>(&self, idx: usize, dim: &D) -> &[T] {
        &self.buf[idx..idx+dim.size()]
    }

    unsafe fn slice_mut<D: Dimension>(&self, idx: usize, dim: &D) -> &mut [T] {
        let slice = self.slice(idx, dim);
        let mut_slice = slice as *const [T] as *mut [T];
        &mut *mut_slice
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        self.buf.clear();
    }
}

impl<T: Float> ArrayCtx<T> for FlatCtx<T> {
    fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_dimension();
        self.empty::<D, D>(dim)
    }

    fn clear(&mut self) {
        self.buf.clear();
    }

    fn id<D: Dimension>(&self, xs: ArrayViewMut<T, D>) -> ArrId<T, D> {
        self.id(xs)
    }

    fn from_id<D: Dimension>(&self, id: &ArrId<T, D>) -> ArrayView<T, D> {
        self.from_id(id)
    }

    fn from_id_mut<D: Dimension>(&self, id: ArrId<T, D>) -> ArrayViewMut<T, D> {
        self.from_id_mut(id)
    }
}
