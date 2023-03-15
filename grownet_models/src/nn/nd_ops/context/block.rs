use super::*;

/// Ctx with memory allocated in blocks, panics if an allocation is bigger than the size of a block
pub struct BlockCtx<T> {
    buf: Vec<Vec<T>>,
    block_size: usize,
    gen: usize,
}

impl<T: Float> BlockCtx<T> {
    pub fn new(block_size: usize) -> Self {
        let buf = Vec::new();
        BlockCtx { buf, block_size, gen: 0 }
    }

    unsafe fn reserve(&self, nelem: usize) -> usize {
        let block_ctx = self as *const BlockCtx<T> as *mut BlockCtx<T>;
        let block_ctx = &mut *block_ctx;
        if nelem > block_ctx.block_size {
            panic!("allocation size is bigger than the block size");
        }
        if block_ctx.buf.len() == 0 || block_ctx.buf.last().unwrap().len() + nelem < block_ctx.buf.last().unwrap().capacity() {
            let mut new_block = Vec::new();
            new_block.reserve_exact(block_ctx.block_size);
            block_ctx.buf.push(new_block);
        }
        let last = block_ctx.buf.last_mut().unwrap();
        let len = last.len();
        last.set_len(len + nelem);

        len + block_ctx.block_size * (block_ctx.buf.len() - 1)
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
        unsafe { 
            let ptr = xs.as_ptr();
            for (i, block) in self.buf.iter().enumerate() {
                let block_ptr = block.as_ptr();
                let offset = ptr.offset_from(block_ptr);
                if offset >= 0 && offset < self.block_size as isize {
                    return ArrId { dim: xs.raw_dim(), offset: offset as usize + i * self.block_size, gen: self.gen, _data: PhantomData };
                }
            }
            panic!("view does not originate from current context");
        }
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

    unsafe fn slice<D: Dimension>(&self, idx: usize, dim: &D) -> &[T] {
        let block_id = idx / self.block_size;
        let idx = idx % self.block_size;
        &self.buf[block_id][idx..idx+dim.size()]
    }

    unsafe fn slice_mut<D: Dimension>(&self, idx: usize, dim: &D) -> &mut [T] {
        let slice = self.slice(idx, dim);
        let mut_slice = slice as *const [T] as *mut [T];
        &mut *mut_slice
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        for block in &mut self.buf {
            block.clear();
        }
    }
}

impl<T: Float> ArrayCtx<T> for BlockCtx<T> {
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
