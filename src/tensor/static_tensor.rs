// Unfortunately, rust does not support arthimetic on const generics
// I wanted to index like this: a[i][j][k], by producing a StatTsInternal<'a, T, D-1>
// at each index operation, but unfortunately, this is not possible right now

pub struct StatTensor<T, const D: usize> {
    pub ptr:     NonNull<T>,
    pub dims:    [usize; D],
    pub strides: [usize; D],
    pub nelems:  usize,
    pub marker:  PhantomData<T>,
}

impl<T, const D: usize> StatTensor<T, D> {
    pub fn new(dims: [usize; D]) -> StatTensor<T, D> {
        let mut strides: [usize; D] = [0; D];
        strides[D-1] = 1;
        for i in (0..D-1).rev() {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
        let nelems = strides[0] * dims[0];
        assert!(nelems > 0, "cannot make 0 sized tensor");

        let layout = Layout::array::<T>(nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };

        StatTensor{ptr, dims, strides, nelems, marker: PhantomData}
    }

    fn index_params(&self) -> ind::StatArr<D> {
        ind::StatArr { dims: self.dims.as_ptr(), strides: self.strides.as_ptr() }
    }
}


struct StatTsInternal<'a, T, const D: usize> {
    pub ptr:     *const T,
    pub dims:    *const usize,
    pub strides: *const usize,
    pub marker:  PhantomData<&'a T>,
}
