use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut, AddAssign, Index, Add};
use std::ptr::{self, NonNull};
use std::mem;
use std::num::Wrapping;
use num::Float;
use rand::{self, thread_rng};

use ndarray::{prelude::*, Shape, StrideShape, ViewRepr, Data, LinalgScalar, linalg, RawData, IndexLonger, ShapeError, ErrorKind};
use rand_distr::{Normal, Distribution, StandardNormal};

/// ParamId logically owns a particular Array in memory
/// but it is only possible to get a reference to it via
/// a reference to the parent allocator.
/// TODO: obtain an unique id corresponding to each allocator, to avoid using the same
///       ParamId on multiple allocators. (only for --debug profiles?)
pub struct ParamId<Sh> {
    idx: usize,
    shape: Sh
}

/// VolatileId which is only valid for a single 'generation', before the parent
/// allocator calls clear, and after its own allocation.
/// Assumes that there will not be usize::MAX number of generations at once.
pub struct VolatileId<Sh> {
    id: ParamId<Sh>,
    generation: Wrapping<usize>,
}


/// Allocates Arrays all packed contiguously, of the same dimension.
/// Each allocation produces a ParamId, which acts as if it logically
/// owns the data.
/// Since each id owns the data, getting the immutable reference is no
/// problem. To mitigate the annoying borrow checker, we use interior mutability
/// such that ParamMutRef is an immutable borrow from ParameterAllocator,
/// but it is possible to get a mutable reference from ParamMutRef, since it
/// logically owns the data.
/// We make it only possible to retrieve the data with a ParamId, since it logically
/// owns the data, it cannot be clonable
pub struct StaticAllocator<T> {
    ptr: NonNull<T>,
    offsets: Vec<usize>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>
}

use ndarray::IntoDimension;


impl<T> StaticAllocator<T> {
    pub fn new() -> Self {
        assert!(mem::size_of::<T>() != 0, "We're not ready to handle ZSTs");
        Self { 
            ptr: NonNull::dangling(), 
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

    /// Reserves/allocates the block of memory specified by dim, returns
    /// the ParamId of that block, and a mutable slice of that specific block.
    /// This is mostly for later initialization schemes, which may
    /// want to specialize to to be a number type, so one can mutably
    /// initialize the memory from the slice.
    /// This function is unsafe, as it makes it possible to alias memory
    unsafe fn alloc<D, Sh>(&mut self, dim: Sh) -> (ParamId<StrideShape<D>>, &mut [T])
        where D: Dimension, Sh: Into<StrideShape<D>> 
    {
        let shape: StrideShape<D> = dim.into();
        let elems = shape.size();
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }
        let id = self.offsets.len();

        self.offsets.push(self.len);

        self.len += elems;

        let id = ParamId { idx: id, shape };

        let slice = unsafe {
            &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().add(self.len - elems), elems)
        };

        (id, slice)
    }

    /// reserves an unitialized chunk of memory with specified dimension, produces an id
    /// denoting the location of that chunk
    pub fn alloc_uninit<D, Sh>(&mut self, dim: Sh) -> ParamId<StrideShape<D>>
    where D: Dimension, Sh: Into<StrideShape<D>> 
    {
        let shape: StrideShape<D> = dim.into();
        let elems = shape.size();
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }
        let id = self.offsets.len();

        self.offsets.push(self.len);

        self.len += elems;

        ParamId { idx: id, shape }
    }

    /// gets a view of the internal array from the id
    pub fn get<'a, D: Dimension>(&self, id: &'a ParamId<StrideShape<D>>) -> ArrayView<'a, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            let view = ArrayView::from_shape_ptr(id.shape.clone(), ptr as *const T);
            view
        }
    }

    /// the mutable version of get, but consumes the id ensuring that
    /// there is only one mutable reference
    pub fn get_mut<'a, D: Dimension>(&self, id: &'a mut ParamId<StrideShape<D>>) -> ArrayViewMut<'a, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            let view = ArrayViewMut::from_shape_ptr(id.shape.clone(), ptr);
            view
        }
    }

    
}


impl<T: Clone> StaticAllocator<T> {
    pub fn alloc_fn<D: Dimension, Sh>(&mut self, dim: Sh, mut f: impl FnMut() -> T) -> ParamId<StrideShape<D>>
        where Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        slice.iter_mut().for_each(|x| {
            *x = f()
        });
        id
    }
    pub fn store<'a, D: Dimension>(&mut self, arr: &ArrayView<'a, T, D>) -> ParamId<StrideShape<D>> {
        let mut id = self.alloc_uninit(arr.dim());
        let mut new_arr = self.get_mut(&mut id);
        new_arr.zip_mut_with(arr, |a, b| {
            *a = b.clone();
        });

        id
    }
}

use ndarray::DimMax;

impl<T> StaticAllocator<T> 
where T: Float, StandardNormal: Distribution<T> {
    pub fn alloc_rand<D, Sh>(&mut self, dim: Sh, mu: T, sigma: T) -> ParamId<StrideShape<D>>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }

    pub fn alloc_randn<D, Sh>(&mut self, dim: Sh) -> ParamId<StrideShape<D>>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }
    
    pub fn binop<D1, D2>(&mut self, a: &ParamId<StrideShape<D1>>, b: &ParamId<StrideShape<D2>>) 
    where D1: Dimension, D2: Dimension + DimMax<D1> {
        let a_len = a.shape.size();
        let b_len = b.shape.size();

        let dim3 = broadcast(a.shape.raw_dim(), b.shape.raw_dim()).unwrap();
        let c = Array::<f32, _>::zeros(dim3);
        let a = self.get(a);
        let b = self.get(b);

        if a_len > b_len {

        }
    }
}

impl<T> Drop for StaticAllocator<T> {
    fn drop(&mut self) {
        unsafe{ free(&mut self.ptr, self.cap); }
    }
}

impl<T> Default for StaticAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}



/// Single threaded buffer holding a large number of tensors of the same dimension.
/// Unlike ParamAllocator, it is possible to free memory with this Allocator,
/// but only all at once.
/// This is used to store intermediate calculations in a single forward and backward pass,
/// which does not need to be retained across training steps.
/// TODO: Disable runtime check for generation correctness for release profile.
pub struct VolatileAllocator<T> {
    data: StaticAllocator<T>,
    global_generation: Wrapping<usize>
}

impl<T> VolatileAllocator<T> {
    pub fn new() -> Self {
        Self { data: StaticAllocator::<T>::new(), global_generation: Wrapping(0) }
    }

    fn grow(&mut self, size: usize) {
        self.data.grow(size);
    }

}

impl<T> VolatileAllocator<T> {
    pub fn alloc_uninit<D, Sh>(&mut self, dim: Sh) -> VolatileId<StrideShape<D>>
        where D: Dimension, Sh: Into<StrideShape<D>> 
    {
        let id = self.data.alloc_uninit(dim);
        VolatileId { id, generation: self.global_generation }
    }

    /// gets a view of the internal array 
    pub fn get<'a, D: Dimension>(&self, id: &'a VolatileId<StrideShape<D>>) -> Option<ArrayView<'a, T, D>> {
        if id.generation != self.global_generation {
            None
        } else {
            Some(self.data.get(&id.id))
        }
    }

    pub fn get_mut<'a, D: Dimension>(&self, id: &'a mut VolatileId<StrideShape<D>>) -> Option<ArrayViewMut<'a, T, D>> {
        if id.generation != self.global_generation {
            return None;
        }
        let view = self.data.get_mut(&mut id.id);
        Some(view)
    }

    pub fn clear(&mut self) {
        self.global_generation += 1;
        self.data.len = 0;
        self.data.offsets.clear();
    }
}


impl<T> Default for VolatileAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Default)]
pub struct ModelAllocatorV1<T> {
    params: StaticAllocator<T>,
    temp_vars: VolatileAllocator<T>
}

impl<T: Float + LinalgScalar> ModelAllocatorV1<T> {

}



#[test]
fn test_static_alloc() {
    let mut alloc = StaticAllocator::<f32>::new();
    let mut a = alloc.alloc_randn([1, 4]);
    let mut b = alloc.alloc_randn([4, 4]);

    let c = alloc.get(&a);
    let mut d = alloc.get_mut(&mut b);
    
    let j = c.broadcast([4, 4]).unwrap();
    
    assert!(j.shape() == d.shape() && d.ndim() == j.ndim());
    //println!("c {}, e {}", *c.0, *e.0);
    
}

use std::iter::IntoIterator;
#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError::from_kind(k)
}
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in out.slice_mut().iter_mut().zip(shape1.slice().iter()) {
        *out = *s;
    }
    let h = out.slice_mut();
    let it = IntoIterator::into_iter(h);
    for (out, s2) in it.zip(shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

pub fn broadcast<D1, D2>(shape1: &D1, shape2: &D2) -> Result<<D2 as DimMax<D1>>::Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension + DimMax<D1>,
    //<D2 as DimMax<D1>>::Output: Into<StrideShape<<D2 as DimMax<D1>>::Output>>
{
    co_broadcast::<D1, D2, <D2 as DimMax<D1>>::Output>(shape1, shape2)
}

#[test]
fn test_broadcast() {
    let a1 = [1usize, 5, 1];
    let a2 = [4usize, 1, 1, 2];
    let h1: Ix3 = a1.into_dimension();
    let h2 = a2.into_dimension();

    let h3 = broadcast(&h1, &h2).unwrap();

    println!("{:?}", h3);
}

/// Simple parameter allocator, mainly used for collecting gradients
/// does not enforce memory safety or the rust memory model
pub struct ParamAllocator<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>
}

impl<T> ParamAllocator<T> {
    pub fn new() -> Self {
        assert!(mem::size_of::<T>() != 0, "We're not ready to handle ZSTs");
        Self { 
            ptr: NonNull::dangling(), 
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

    fn alloc(&mut self, elems: usize) -> &mut [T] {
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }

        let slice = unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().add(self.len), elems) };
        self.len += elems;
        slice
    }

    /// reserves an unitialized chunk of memory with specified dimension, produces an id
    /// denoting the location of that chunk
    pub fn alloc_uninit<D, Sh>(&mut self, dim: Sh) -> OwnedParam<T, D>
        where D: Dimension, Sh: Into<StrideShape<D>>
    {
        let shape: StrideShape<D> = dim.into();
        let elems = shape.size();
        if elems + self.len >= self.cap {
            self.grow(elems + self.len - self.cap);
        }
        let new_ptr = unsafe {
            NonNull::new(self.ptr.as_ptr().add(self.len)).unwrap()
        };

        let param = OwnedParam { ptr: new_ptr, shape: shape };
        self.len += elems;
        param
    }
}

impl<T: Clone> ParamAllocator<T> {
    pub fn alloc_fn<D, Sh>(&mut self, dim: Sh, mut f: impl FnMut() -> T) -> OwnedParam<T, D>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        
        let shape: StrideShape<D> = dim.into();
        let slice = self.alloc(shape.size());
        slice.iter_mut().for_each(|x| {
            *x = f()
        });

        OwnedParam { ptr: NonNull::new(slice.as_mut_ptr()).unwrap(), shape: shape }
    }
}

impl<T> ParamAllocator<T> 
where T: Float, StandardNormal: Distribution<T> {
    pub fn alloc_rand<D, Sh>(&mut self, dim: Sh, mu: T, sigma: T) -> OwnedParam<T, D>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        let mut rng = thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();

        self.alloc_fn(dim, || {
            normal.sample(&mut rng)
        })
    }

    pub fn alloc_randn<D, Sh>(&mut self, dim: Sh) -> OwnedParam<T, D>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        self.alloc_rand(dim, T::zero(), T::one())
    }
}

impl<T> Drop for ParamAllocator<T> {
    fn drop(&mut self) {
        unsafe{ free(&mut self.ptr, self.cap); }
    }
}

impl<T> Default for ParamAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OwnedParam<T, D> {
    ptr: NonNull<T>,
    shape: StrideShape<D>
}

impl<T, D: Dimension> OwnedParam<T, D> {
    fn get(&self) -> ArrayView<'_, T, D> {
        unsafe{ ArrayView::from_shape_ptr(self.shape.clone(), self.ptr.as_ptr()) }
    }
    fn get_mut(&mut self) -> ArrayViewMut<'_, T, D> {
        unsafe{ ArrayViewMut::from_shape_ptr(self.shape.clone(), self.ptr.as_ptr()) }
    }
}

impl<T: Float, D: Dimension> OwnedParam<T, D> {
    fn zero(&mut self) {
        for i in 0..self.shape.size() {
            unsafe {
                *self.ptr.as_ptr().add(i) = T::zero();
            }
        }
    }
}

struct ModelAllocator {
    params: ParamAllocator<f32>,
    grads: ParamAllocator<f32>
}

impl ModelAllocator {
    fn alloc_randn<D, Sh>(&mut self, dim: Sh) -> (OwnedParam<f32, D>, OwnedParam<f32, D>)
    where D: Dimension, Sh: Into<StrideShape<D>> + Clone {
        (self.params.alloc_randn(dim.clone()), self.grads.alloc_fn(dim, || 0.0))
    }
}

pub struct Linear {
    w: OwnedParam<f32, Ix2>,
    b: OwnedParam<f32, Ix2>,
    dw: OwnedParam<f32, Ix2>,
    db: OwnedParam<f32, Ix2>,
}


impl Linear {
    fn init(allocator: &mut ModelAllocator, dim: usize) -> Self {
        let (w, dw) = allocator.alloc_randn([dim, dim]);
        let (b, db) = allocator.alloc_randn([1, dim]);
        Self { w, b, dw, db }
    }

    pub fn forward(&self, data: Array2<f32>) -> Array2<f32> {
        
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

/*
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
*/