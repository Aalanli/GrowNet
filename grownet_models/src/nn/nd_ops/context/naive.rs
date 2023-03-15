use std::{cell::RefCell, ops::AddAssign};

use super::*;

/// Does not pool anything, mainly used to test how much memory something is using
pub struct NaiveCtx<T> {
    buf: RefCell<Vec<Vec<T>>>,
    allocated: RefCell<usize>,
    gen: usize,
}

impl<T: Float> NaiveCtx<T> {
    pub fn new() -> Self {
        let buf = Vec::new();
        NaiveCtx { buf: RefCell::new(buf), gen: 0, allocated: RefCell::new(0) }
    }

    fn reserve(&self, nelem: usize) -> usize {
        let blocks = &mut *self.buf.borrow_mut();
        let mut buf = Vec::new();
        buf.reserve_exact(nelem);
        unsafe {
            buf.set_len(nelem);
        }
        blocks.push(buf);
        self.allocated.borrow_mut().add_assign(nelem);
        blocks.len() - 1
    }
    
    /// panics if there is not enough space
    pub fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_shape();
        let raw_dim = dim.raw_dim();
        let nelem = dim.size();
        let idx = self.reserve(nelem);
        let xs = unsafe{ self.slice_mut(idx, raw_dim) };
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
        let ptr = xs.as_ptr();
        for (i, block) in self.buf.borrow().iter().enumerate() {
            let block_ptr = block.as_ptr();
            if ptr == block_ptr {
                return ArrId { dim: xs.raw_dim(), offset: i, gen: self.gen, _data: PhantomData };
            }
        }
        panic!("view does not originate from current context");
    }

    pub fn from_id<D: Dimension>(&self, id: &ArrId<T, D>) -> ArrayView<T, D> {
        assert!(id.gen == self.gen, "generation mismatch between id and ctx");
        let arr = unsafe {
            let ptr = self.slice(id.offset, &id.dim).as_ptr();
            ArrayView::from_shape_ptr(id.dim.clone(), ptr)
        };
        arr
    }

    pub fn from_id_mut<D: Dimension>(&self, id: ArrId<T, D>) -> ArrayViewMut<T, D> {
        assert!(id.gen == self.gen, "generation mismatch between id and ctx");
        let arr = unsafe {
            let ptr = self.slice_mut(id.offset, &id.dim).as_mut_ptr();
            ArrayViewMut::from_shape_ptr(id.dim.clone(), ptr)
        };
        arr
    }

    unsafe fn slice<D: Dimension>(&self, idx: usize, _dim: &D) -> &[T] {
        let buf = self.buf.borrow();
        let buf = buf.get(idx).unwrap().as_slice();
        &*(buf as *const [T])
    }

    unsafe fn slice_mut<D: Dimension>(&self, idx: usize, dim: &D) -> &mut [T] {
        let slice = self.slice(idx, dim);
        let mut_slice = slice as *const [T] as *mut [T];
        &mut *mut_slice
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        self.buf.borrow_mut().clear();
    }

    pub fn allocated(&self) -> usize {
        *self.allocated.borrow()
    }
}

impl<T: Float> ArrayCtx<T> for NaiveCtx<T> {
    fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D>>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let dim = dim.into_dimension();
        self.empty::<D, D>(dim)
    }

    fn clear(&mut self) {
        self.buf.borrow_mut().clear();
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
