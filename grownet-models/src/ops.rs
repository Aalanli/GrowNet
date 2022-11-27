use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::mem;
use std::num::Wrapping;
use num::Float;
use rand::{self, thread_rng};

use ndarray::{prelude::*, Shape, StrideShape, ViewRepr, Data, LinalgScalar, linalg, RawData};
use rand_distr::{Normal, Distribution, StandardNormal};


/// Allocates Arrays all packed contiguously, of the same dimension.
/// Each allocation produces a ParamId, which acts as if it logically
/// owns the data.
/// Since each id owns the data, getting the immutable reference is no
/// problem. To mitigate the annoying borrow checker, we use interior mutability
/// such that ParamMutRef is an immutable borrow from ParameterAllocator,
/// but it is possible to get a mutable reference from ParamMutRef, since it
/// logically owns the data.
/// We make it only possible to retrieve the data with a ParamId, since it logically
/// owns the data, it cannot be clonable. We make it impossible to get two
/// ParamMutRefs at the same time, as making a ParamMutRef consumes the only copy
/// of ParamId. It is only possible to get that copy back by destroying/consuming
/// a ParamMutRef.
pub struct StaticAllocator<T, D> {
    ptr: NonNull<T>,
    shapes: Vec<StrideShape<D>>,
    offsets: Vec<usize>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>
}

impl<T, D> StaticAllocator<T, D> {
    pub fn new() -> Self {
        assert!(mem::size_of::<T>() != 0, "We're not ready to handle ZSTs");
        Self { 
            ptr: NonNull::dangling(), 
            shapes: Vec::new(), 
            offsets: Vec::new(),
            len: 0, cap: 0, 
            _marker: PhantomData 
        }
    }
    /// grows the internal storage in addition to the size that it has
    pub fn grow(&mut self, size: usize) {
        grow::<T>(size, &mut self.ptr, &mut self.cap);
    }

    /// current number of elements allocated, useful for future mass allocation/reservation
    pub fn capacity(&self) -> usize { self.cap }

    /// Gets a linear slice to the entire block of memory that is allocated
    pub fn slice(&self) -> &[T] {
        unsafe { &*ptr::slice_from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T, D: Dimension> StaticAllocator<T, D> {
    /// Reserves/allocates the block of memory specified by dim, returns
    /// the ParamId of that block, and a mutable slice of that specific block.
    /// This is mostly for later initialization schemes, which may
    /// want to specialize to to be a number type, so one can mutably
    /// initialize the memory from the slice.
    /// This function is unsafe, as it makes it possible to alias memory
    unsafe fn alloc<Sh>(&mut self, dim: Sh) -> (ParamId, &mut [T])
        where Sh: Into<StrideShape<D>> 
    {
        let shape: StrideShape<D> = dim.into();
        let elems = shape.size();
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }
        let id = self.offsets.len();

        self.shapes.push(shape);
        self.offsets.push(self.len);

        self.len += elems;

        let id = ParamId { idx: id };

        let slice = unsafe {
            &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().add(self.len - elems), elems)
        };

        (id, slice)
    }

    /// reserves an unitialized chunk of memory with specified dimension, produces an id
    /// denoting the location of that chunk
    pub fn alloc_uninit<Sh>(&mut self, dim: Sh) -> ParamId
        where Sh: Into<StrideShape<D>> 
    {
        let shape: StrideShape<D> = dim.into();
        let elems = shape.size();
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }
        let id = self.offsets.len();

        self.shapes.push(shape);
        self.offsets.push(self.len);

        self.len += elems;

        ParamId { idx: id }
    }

    /// gets a view of the internal array from the id
    pub fn get(&self, id: &ParamId) -> ArrayView<'_, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            ArrayView::from_shape_ptr(self.shapes[id.idx].clone(), ptr as *const T)
        }
    }

    /// the mutable version of get, but consumes the id ensuring that
    /// there is only one mutable reference
    pub fn get_mut(&self, id: ParamId) -> ParamMutRef<'_, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            let view = ArrayViewMut::from_shape_ptr(self.shapes[id.idx].clone(), ptr);
            ParamMutRef { id, view }
        }
    }

    /// This is especially useful for computing statistics of internal arrays
    pub fn iter(&self) -> impl Iterator<Item = ArrayView<'_, T, D>> {
        (0..self.shapes.len()).map(|x| {
            let id = ParamId { idx: x };
            self.get(&id)
        })
    }

    /// this seems like it would violate the contract of ParamId, but there is a 
    /// subtle difference between actually owning the data, and what ParamId does, since
    /// ParamId can only get the data it 'owns' with a reference to this allocator.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ArrayViewMut<'_, T, D>> {
        (0..self.shapes.len()).map(|x| {
            unsafe {
                let ptr = self.ptr.as_ptr().add(self.offsets[x]);
                let view = ArrayViewMut::from_shape_ptr(self.shapes[x].clone(), ptr);
                view
            }
        })
    }
}

impl<T: Clone, D: Dimension> StaticAllocator<T, D> {
    pub fn alloc_fn<Sh>(&mut self, dim: Sh, mut f: impl FnMut() -> T) -> ParamId
        where Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        slice.iter_mut().for_each(|x| {
            *x = f()
        });
        id
    }
}

impl<T, D: Dimension> StaticAllocator<T, D> 
where T: Float, StandardNormal: Distribution<T> {
    pub fn alloc_rand<Sh>(&mut self, dim: Sh, mu: T, sigma: T) -> ParamId
        where Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }

    pub fn alloc_randn<Sh>(&mut self, dim: Sh) -> ParamId
        where Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }
}

impl<T, D> Drop for StaticAllocator<T, D> {
    fn drop(&mut self) {
        unsafe{ free(&mut self.ptr, self.cap); }
    }
}

impl<T, D> Default for StaticAllocator<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// ParamId logically owns a particular Array in memory
/// but it is only possible to get a reference to it via
/// a reference to the parent allocator.
/// TODO: obtain an unique id corresponding to each allocator, to avoid using the same
///       ParamId on multiple allocators. (only for --debug profiles?)
pub struct ParamId {
    idx: usize
}

pub struct ParamMutRef<'a, T, D> {
    id: ParamId,
    view: ArrayViewMut<'a, T, D>
}

impl<'a, T, D> ParamMutRef<'a, T, D> {
    fn get(&mut self) -> &mut ArrayViewMut<'a, T, D> {
        &mut self.view
    }

    fn id(self) -> ParamId {
        self.id
    }
}


/// Single threaded buffer holding a large number of tensors of the same dimension.
/// Unlike ParamAllocator, it is possible to free memory with this Allocator,
/// but only all at once.
/// This is used to store intermediate calculations in a single forward and backward pass,
/// which does not need to be retained across training steps.
/// TODO: Disable runtime check for generation correctness for release profile.
pub struct VolatileAllocator<T, D> {
    data: StaticAllocator<T, D>,
    global_generation: Wrapping<usize>
}

impl<T, D> VolatileAllocator<T, D> {
    pub fn new() -> Self {
        Self { data: StaticAllocator::<T, D>::new(), global_generation: Wrapping(0) }
    }

    fn grow(&mut self, size: usize) {
        self.data.grow(size);
    }

}

impl<T, D: Dimension> VolatileAllocator<T, D> {
    pub fn alloc_uninit<Sh>(&self, dim: Sh) -> VolatileId
        where Sh: Into<StrideShape<D>> 
    {
        let id = unsafe {
            let ptr = &self.data as *const StaticAllocator<T, D> as *mut StaticAllocator<T, D>;
            (*ptr).alloc_uninit(dim)
        };
        VolatileId { id, generation: self.global_generation }
    }

    /// gets a view of the internal array 
    pub fn get(&self, id: &VolatileId) -> Option<ArrayView<'_, T, D>> {
        if id.generation != self.global_generation {
            None
        } else {
            Some(self.data.get(&id.id))
        }
    }

    pub fn get_mut(&self, id: VolatileId) -> Option<VolatileMutRef<'_, T, D>> {
        if id.generation != self.global_generation {
            return None;
        }
        let view = self.data.get_mut(id.id);
        Some(VolatileMutRef {generation: self.global_generation, view})
    }

    pub fn clear(&mut self) {
        self.global_generation += 1;
        self.data.len = 0;
        self.data.shapes.clear();
        self.data.offsets.clear();
    }
}

impl<T> VolatileAllocator<T, Ix2> 
where T: LinalgScalar {
    /// matmuls a and b, stores the result, referenced to by the return value
    pub fn mat(&mut self, a: &VolatileId, b: &VolatileId) -> Option<VolatileId> {
        let a = self.get(a)?;
        let b = self.get(b)?;
        let adim = a.dim();
        let bdim = b.dim();
        let cdim = [adim.0, bdim.1];
        let c_id = self.alloc_uninit(cdim);
        let mut c = self.get_mut(c_id)?;
        let data = c.get();
        linalg::general_mat_mul(T::one(), &a, &b, T::zero(), data);

        Some(c.id())
    }

    /// matmuls a and b, stores the result, referenced to by the return value
    pub fn mat_arr(&self, a: &ArrayView<'_, T, Ix2>, b: &ArrayView<'_, T, Ix2>) -> VolatileId {
        let adim = a.dim();
        let bdim = b.dim();
        let cdim = [adim.0, bdim.1];
        let c_id = self.alloc_uninit(cdim);
        let mut c = self.get_mut(c_id).unwrap();
        let data = c.get();
        linalg::general_mat_mul(T::one(), &a, &b, T::zero(), data);

        c.id()
    }
    
    pub fn bin_op<F>(&mut self, a: &VolatileId, b: &VolatileId, mut f: F) -> Option<VolatileId> 
    where F: FnMut(&mut ArrayViewMut<'_, T, Ix2>, &ArrayView<'_, T, Ix2>, &ArrayView<'_, T, Ix2>) {
        let a = self.get(a)?;
        let b = self.get(b)?;
        if a.dim() != b.dim() {
            return None;
        }
        let c_id = self.alloc_uninit(a.dim());
        let mut c = self.get_mut(c_id)?;
        {
            let view = c.get();
            f(view, &a, &b)
        }
        Some(c.id())
    }

    pub fn add(&mut self, a: &VolatileId, b: &VolatileId) -> Option<VolatileId> {
        self.bin_op(a, b, |c, a, b| {
            c.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((c, a), b)| {
                *c = *a + *b;
            });
        })
    }
}

impl<T, D> Default for VolatileAllocator<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// VolatileId which is only valid for a single 'generation', before the parent
/// allocator calls clear, and after its own allocation.
/// Assumes that there will not be usize::MAX number of generations at once.
pub struct VolatileId {
    id: ParamId,
    generation: Wrapping<usize>,
}

pub struct VolatileMutRef<'a, T, D> {
    generation: Wrapping<usize>,
    view: ParamMutRef<'a, T, D>
}

impl<'a, T, D: Dimension> VolatileMutRef<'a, T, D> {
    fn get(&mut self) -> &mut ArrayViewMut<'a, T, D> {
        self.view.get()
    }

    fn id(self) -> VolatileId {
        VolatileId { generation: self.generation, id: self.view.id() }
    }
}


#[derive(Default)]
pub struct ModelAllocatorV1 {
    params: StaticAllocator<f32, Ix2>,
    grads:  StaticAllocator<f32, Ix2>,
    temp_vars: VolatileAllocator<f32, Ix2>
}

impl ModelAllocatorV1 {
    pub fn alloc_randn(&mut self, dim: [usize; 2]) -> (ParamId, ParamId) {
        let param = self.params.alloc_randn(dim);
        let grad = self.grads.alloc_fn(dim, || 0.0);
        (param, grad)
    }
}


pub struct Linear {
    w: ParamId,
    b: ParamId,
    dw: ParamId,
    db: ParamId,
}

impl Linear {
    fn init(allocator: &mut ModelAllocatorV1, dim: usize) -> Self {
        let (w, dw) = allocator.alloc_randn([dim, dim]);
        let (b, db) = allocator.alloc_randn([1, dim]);
        Self { w, b, dw, db }
    }

    pub fn forward(&self, allocator: &mut ModelAllocatorV1, data: VolatileId) -> VolatileId {
        let w = allocator.params.get(&self.w);
        let data = allocator.temp_vars.get(&data).unwrap();
        let ty = allocator.temp_vars.mat_arr(&w, &data);
        todo!()
    }
}


#[derive(Default)]
pub struct Relu {}

impl Relu {
    pub fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    pub fn backward(&self, x: f32) -> f32 {
        x.max(0.0).min(1.0)
    }
}


#[derive(Default)]
pub struct Normalize {
    mu: f32,
    std: f32
}

impl Normalize {
    const EPSILON: f32 = 1e-6;
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forward(&mut self, x: ArrayView1<f32>, mut y: ArrayViewMut1<f32>) {
        self.mu = x.iter().fold(0.0, |x, y| x + *y) / x.len() as f32;
        self.std = x.iter().fold(0.0, |x, y| x + (*y - self.mu).powf(2.0));
        self.std /= (x.len() - 1) as f32;
        self.std = self.std.sqrt();

        y.iter_mut().zip(x.iter()).for_each(|(y, x)| {
            *y = (*x - self.mu) / (self.std + Self::EPSILON);
        });
    }

    pub fn backward(&mut self, grad: ArrayView1<f32>, x: ArrayView1<f32>, mut dy_dx: ArrayViewMut1<f32>) {
        let nvar = 1.0 / (self.std + Self::EPSILON);
        let mut dotx = 0.0;
        let mut sum_grad = 0.0;
        grad.iter().zip(x.iter()).for_each(|(x, y)| {
            sum_grad += *x;
            dotx += *x * *y;
        });

        let m = nvar * nvar / (self.std * (x.len() - 1) as f32);
        let m = (sum_grad * self.mu - dotx) / m;
        let b = sum_grad / (x.len() as f32) * nvar;
        grad.iter().zip(x.iter()).zip(dy_dx.iter_mut()).for_each(|((g, x), y)| {
            let x = *x;
            let g = *g;
            *y = (x - self.mu) * m - b + g * nvar;
        });
    }
}

/// grows the specified pointer by size, to cap + size
/// mutably sets the pointer to the new allocated pointer, and cap to the new cap
fn grow<T>(size: usize, ptr: &mut NonNull<T>, cap: &mut usize) {
    let (new_cap, new_layout) = if *cap == 0 {
        (size, Layout::array::<T>(size).unwrap())
    } else {
        let new_cap = *cap + size;
        let new_layout = Layout::array::<T>(new_cap).unwrap();
        (new_cap, new_layout)
    };

    assert!(new_layout.size() <= isize::MAX as usize, "Allocation too large");

    let new_ptr = if *cap == 0 {
        unsafe { alloc::alloc(new_layout) }
    } else {
        let old_layout = Layout::array::<T>(*cap).unwrap();
        let old_ptr = ptr.as_ptr() as *mut u8;
        unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) }
    };
    // If allocation fails, `new_ptr` will be null, in which case we abort.
    *ptr = match NonNull::new(new_ptr as *mut T) {
        Some(p) => p,
        None => alloc::handle_alloc_error(new_layout),
    };
    *cap = new_cap;
}

/// frees the block pointed to by ptr
unsafe fn free<T>(ptr: &mut NonNull<T>, cap: usize) {
    if cap != 0 {
        let ptr = ptr.as_ptr();
        for i in 0..cap {
            ptr::drop_in_place(ptr.add(i));
        }
        let layout = Layout::array::<T>(cap).unwrap();
        unsafe {
            alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

#[test]
fn normalize_grad() {
    use ndarray_rand::{RandomExt, rand_distr::Normal, rand_distr::Uniform};

    let dim = 16;
    let x: Array1<f32> = Array::random((dim,), Normal::new(0.0, 1.0).unwrap());
    let grad: Array1<f32> = Array1::ones([dim]);
    let mut dy_dx: Array1::<f32> = Array1::ones([dim]);
    let mut y: Array1::<f32> = Array1::ones([dim]);
    
    let mut norm = Normalize::new();
    norm.forward(x.view(), y.view_mut());
    norm.backward(grad.view(), x.view(), dy_dx.view_mut());

    println!("input {}", x);
    println!("forward pass {}", y);
    println!("backward pass {}", dy_dx);
}

#[test]
fn parameter_alloc_test() {
    let mut params = StaticAllocator::<f32, Ix2>::new();
    let a = params.alloc_uninit([4, 4]);
    let b = params.alloc_uninit([4, 5]);
    let c = params.alloc_uninit([4, 5]);

    let _a = params.get(&a);
    let _b = params.get(&b);
    let _c = params.get(&c);

    println!("shapes {:?}, cap {}, len {}, offsets {:?}", params.shapes, params.cap, params.len, params.offsets);
}

#[test]
fn volatile_matmul_test() {
    let mut volatile = VolatileAllocator::<f32, Ix2>::new();
    let a = volatile.alloc_uninit([5, 5]);
    let b = volatile.alloc_uninit([5, 4]);
    let c = volatile.mat(&a, &b).unwrap();

    println!("generation {}, result_shape {:?}", c.generation, volatile.get(&c).unwrap().shape());
    println!("a \n{}", volatile.get(&a).unwrap());
}

