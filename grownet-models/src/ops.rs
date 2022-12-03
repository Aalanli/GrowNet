use std::fmt::{Display, Debug};
use std::marker::PhantomData;
use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut, AddAssign, Index, Add};
use std::ptr::{self, NonNull};
use std::mem::{self, MaybeUninit};
use std::num::Wrapping;
use ndarray::linalg::general_mat_mul;
use num::Float;
use rand::{self, thread_rng};

use ndarray::{prelude::*, IntoDimension, DimMax, Shape, StrideShape, ViewRepr, Data, LinalgScalar, linalg, RawData, IndexLonger, ShapeError, ErrorKind};
use rand_distr::{Normal, Distribution, StandardNormal};

/// ParamId logically owns a particular Array in memory
/// but it is only possible to get a reference to it via
/// a reference to the parent allocator.
/// TODO: obtain an unique id corresponding to each allocator, to avoid using the same
///       ParamId on multiple allocators. (only for --debug profiles?)
#[derive(Debug)]
pub struct ParamId<D> {
    idx: usize,
    shape: StrideShape<D>
}

/// VolatileId which is only valid for a single 'generation', before the parent
/// allocator calls clear, and after its own allocation.
/// Assumes that there will not be usize::MAX number of generations at once.
#[derive(Debug)]
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
pub struct ArrayAllocator<T> {
    ptr: NonNull<T>,
    offsets: Vec<usize>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>
}


impl<T> ArrayAllocator<T> {
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
        let new_size = ((size * 50) as f32 + (self.cap as f32) * 0.7) as usize;
        self.ptr = grow::<T>(new_size, self.ptr, self.cap);
        self.cap += new_size;
    }

    /// current number of elements allocated, useful for future mass allocation/reservation
    pub fn capacity(&self) -> usize { self.cap }

    /// Gets a linear slice to the entire block of memory that is allocated
    pub unsafe fn slice(&self) -> &[T] {
        unsafe { &*ptr::slice_from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub unsafe fn slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Reserves/allocates the block of memory specified by dim, returns
    /// the ParamId of that block, and a mutable slice of that specific block.
    /// This is mostly for later initialization schemes, which may
    /// want to specialize to to be a number type, so one can mutably
    /// initialize the memory from the slice.
    /// This function is unsafe, as it makes it possible to alias memory
    unsafe fn alloc<D, Sh>(&mut self, dim: Sh) -> (ParamId<D>, & mut [T])
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
    pub fn alloc_uninit<D, Sh>(&mut self, dim: Sh) -> ParamId<D>
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
    /// This is unsafe because the underlying memory pointer could change if any allocation opeations lie between
    /// a get and another get. One must do all allocations before get operations, otherwise, returned references could
    /// be pointing to invalid memory.
    pub unsafe fn get<'a, D: Dimension>(&self, id: &'a ParamId<D>) -> ArrayView<'a, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            let view = ArrayView::from_shape_ptr(id.shape.raw_dim().clone(), ptr as *const T);
            view
        }
    }

    /// the mutable version of get, but consumes the id ensuring that
    /// there is only one mutable reference
    pub unsafe fn get_mut<'a, D: Dimension>(&self, id: &'a mut ParamId<D>) -> ArrayViewMut<'a, T, D> {
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.offsets[id.idx]);
            let view = ArrayViewMut::from_shape_ptr(id.shape.raw_dim().clone(), ptr);
            view
        }
    }
}



impl<T: Clone> ArrayAllocator<T> {
    pub fn get_copy<'a, D: Dimension>(&self, id: &'a ParamId<D>) -> Array<T, D> {
        unsafe { self.get(id).to_owned() }
    }

    pub fn alloc_fn<D: Dimension, Sh>(&mut self, dim: Sh, mut f: impl FnMut() -> T) -> ParamId<D>
        where Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        slice.iter_mut().for_each(|x| {
            *x = f()
        });
        id
    }
    
    pub fn store<'a, D: Dimension>(&mut self, arr: &ArrayView<'a, T, D>) -> ParamId<D> {
        let mut id = self.alloc_uninit(arr.dim());
        let mut new_arr = unsafe{ self.get_mut(&mut id) };
        new_arr.zip_mut_with(arr, |a, b| {
            *a = b.clone();
        });

        id
    }
}

impl<T> ArrayAllocator<T> 
where T: Float + Debug, StandardNormal: Distribution<T> {
    pub fn alloc_rand<D, Sh>(&mut self, dim: Sh, mu: T, sigma: T) -> ParamId<D>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }

    pub fn alloc_randn<D, Sh>(&mut self, dim: Sh) -> ParamId<D>
        where D: Dimension, Sh: Into<StrideShape<D>> {
        let (id, slice) = unsafe{ self.alloc(dim) };
        let mut rng = thread_rng();
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        slice.iter_mut().for_each(|x| {
            *x = normal.sample(&mut rng);
        });
        id
    }
    
    pub fn binop<D1, D2>(&mut self, a: &ParamId<D1>, b: &ParamId<D2>, mut f: impl FnMut(T, T) -> T) 
        -> ParamId<<D2 as DimMax<D1>>::Output>
    where D1: Dimension, D2: Dimension + DimMax<D1> {
        let dim3 = broadcast(a.shape.raw_dim(), b.shape.raw_dim()).unwrap();
        let mut c_id = self.alloc_uninit(dim3.clone());

        unsafe {
            let mut c = self.get_mut(&mut c_id);
            let a = self.get(a);
            let b = self.get(b);

            let mut itercount = 0;
            
            c.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((a, b), c)| {*a = f(*b, *c); itercount += 1;});
            
            c_id
        }
    }

    pub fn x<'a, 'b, D>(&'a mut self, id: &'b ParamId<D>) -> AllocOps<'a, ArrayView<'b, T, D>, T> 
    where D: Dimension {
        let view = unsafe { self.get(id) };
        AllocOps { alloc: self, id: view }
    }

    pub fn xmut<'a, 'b, D>(&'a mut self, id: &'b mut ParamId<D>) -> AllocOps<'a, ArrayViewMut<'b, T, D>, T> 
    where D: Dimension {
        let view = unsafe { self.get_mut(id) };
        AllocOps { alloc: self, id: view }
    }

    pub fn xs<'a, 'b, D, const N: usize>(&'a mut self, ids: [&'b ParamId<D>; N])
        -> AllocOps<'a, [ArrayView<'b, T, D>; N], T>
    where D: Dimension + Sized, T: Sized
    {
        let view = unsafe {
            let mut arr: [MaybeUninit<ArrayView<T, D>>; N] = MaybeUninit::uninit().assume_init();
            arr.iter_mut().zip(ids.iter()).for_each(|(v, id)| 
                {*v = MaybeUninit::new(self.get(id));}
            );

            transmute_arr(arr)
        };
        AllocOps { alloc: self, id: view }
    }

    pub fn xsmut<'a, 'b, D, const N: usize>(&'a mut self, mut ids: [&'b mut ParamId<D>; N])
        -> AllocOps<'a, [ArrayViewMut<'b, T, D>; N], T>
    where D: Dimension + Sized, T: Sized
    {
        let view = unsafe {
            let mut arr: [MaybeUninit<ArrayViewMut<T, D>>; N] = MaybeUninit::uninit().assume_init();
            arr.iter_mut().zip(ids.iter_mut()).for_each(|(v, id)| 
                {*v = MaybeUninit::new(self.get_mut(*id));}
            );

            let ptr = &arr as *const [MaybeUninit<ArrayViewMut<T, D>>; N];
            let ptr = ptr as *const [ArrayViewMut<T, D>; N];
            let val = ptr.read();
            mem::forget(arr);
            val
        };
        AllocOps { alloc: self, id: view }
    }
    
    pub fn zero_<D: Dimension>(&self, a: &mut ParamId<D>) {
        let mut a = unsafe { self.get_mut(a) };
        a.map_mut(|v| { *v = T::zero(); } );
    }
}

pub struct AllocOps<'a, P, T> {
    alloc: &'a mut ArrayAllocator<T>,
    id: P
}

impl<'a, P, T> AllocOps<'a, P, T> {
    pub fn op(self, f: impl Fn(P)) {
        f(self.id)
    }
}

impl<'a, P, T> AllocOps<'a, P, T>
{
    pub fn xs<'b, D, const N: usize>(self, ids: [&'b ParamId<D>; N])
        -> AllocOps<'a, (P, [ArrayView<'b, T, D>; N]), T>
    where D: Dimension + Sized, T: Sized
    {
        let view = unsafe {
            let mut arr: [MaybeUninit<ArrayView<T, D>>; N] = MaybeUninit::uninit().assume_init();
            arr.iter_mut().zip(ids.iter()).for_each(|(v, id)| 
                {*v = MaybeUninit::new(self.alloc.get(id));}
            );

            transmute_arr(arr)
        };
        AllocOps { alloc: self.alloc, id: (self.id, view) }
    }

    pub fn xsmut<'b, D, const N: usize>(self, mut ids: [&'b mut ParamId<D>; N])
        -> AllocOps<'a, (P, [ArrayViewMut<'b, T, D>; N]), T>
    where D: Dimension + Sized, T: Sized
    {
        let view = unsafe {
            let mut arr: [MaybeUninit<ArrayViewMut<T, D>>; N] = MaybeUninit::uninit().assume_init();
            arr.iter_mut().zip(ids.iter_mut()).for_each(|(v, id)| 
                {*v = MaybeUninit::new(self.alloc.get_mut(*id));}
            );

            let ptr = &arr as *const [MaybeUninit<ArrayViewMut<T, D>>; N];
            let ptr = ptr as *const [ArrayViewMut<T, D>; N];
            let val = ptr.read();
            mem::forget(arr);
            val
        };
        AllocOps { alloc: self.alloc, id: (self.id, view) }
    }

    pub fn x<'b, D>(self, id: &'b ParamId<D>) 
        -> AllocOps<'a, (P, ArrayView<'b, T, D>), T>
    where D: Dimension
    {
        let view = unsafe { self.alloc.get(id) };
        AllocOps { alloc: self.alloc, id: (self.id, view) }
    }

    pub fn xmut<'b, D>(self, id: &'b mut ParamId<D>) 
        -> AllocOps<'a, (P, ArrayViewMut<'b, T, D>), T>
    where D: Dimension
    {
        let view = unsafe { self.alloc.get_mut(id) };
        AllocOps { alloc: self.alloc, id: (self.id, view) }
    }
}



impl<T: LinalgScalar> ArrayAllocator<T> {
    pub fn matmul(&mut self, a: &ParamId<Ix2>, b: &ParamId<Ix2>) {
        let sh_a = a.shape.raw_dim()[0];
        let sh_b = b.shape.raw_dim()[1];
        let mut c_id = self.alloc_uninit([sh_a, sh_b]);
        unsafe {
            let a = self.get(a);
            let b = self.get(b);
            let mut c = self.get_mut(&mut c_id);
            general_mat_mul(T::one(), &a, &b, T::zero(), &mut c);
        }
    }
}

impl<T> Drop for ArrayAllocator<T> {
    fn drop(&mut self) {
        unsafe{ free(&mut self.ptr, self.cap); }
    }
}

impl<T> Default for ArrayAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}


pub struct Linear {
    w: ParamId<Ix2>,
    b: ParamId<Ix2>,
    dw: ParamId<Ix2>,
    db: ParamId<Ix2>,
}


impl Linear {
    fn init(allocator: &mut ArrayAllocator<f32>, dim: usize) -> Self {
        let w = allocator.alloc_randn([dim, dim]);
        let dw = allocator.alloc_randn([dim, dim]);
        let b = allocator.alloc_randn([1, dim]);
        let db = allocator.alloc_randn([1, dim]);
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
fn grow<T>(size: usize, ptr: NonNull<T>, cap: usize) -> NonNull<T> {
    let (_new_cap, new_layout) = if cap == 0 {
        (size, Layout::array::<T>(size).unwrap())
    } else {
        let new_cap = cap + size;
        let new_layout = Layout::array::<T>(new_cap).unwrap();
        (new_cap, new_layout)
    };

    assert!(new_layout.size() <= isize::MAX as usize, "Allocation too large");

    let new_ptr = if cap == 0 {
        unsafe { alloc::alloc(new_layout) }
    } else {
        let old_layout = Layout::array::<T>(cap).unwrap();
        let old_ptr = ptr.as_ptr() as *mut u8;
        unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) }
    };
    // If allocation fails, `new_ptr` will be null, in which case we abort.
    match NonNull::new(new_ptr as *mut T) {
        Some(p) => p,
        None => alloc::handle_alloc_error(new_layout),
    }
}

#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError::from_kind(k)
}

/// Helper broadcasting function copied from ndarray source
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (_k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
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

/// broadcasts two dimensions for binary operations
pub fn broadcast<D1, D2>(shape1: &D1, shape2: &D2) -> Result<<D2 as DimMax<D1>>::Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension + DimMax<D1>,
    //<D2 as DimMax<D1>>::Output: Into<StrideShape<<D2 as DimMax<D1>>::Output>>
{
    co_broadcast::<D1, D2, <D2 as DimMax<D1>>::Output>(shape1, shape2)
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


unsafe fn transmute_arr<T, const N: usize>(arr: [MaybeUninit<T>; N]) -> [T; N] {
    let ptr = &arr as *const [MaybeUninit<T>; N] as *const [T; N];
    let val = ptr.read();
    mem::forget(arr);
    val
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
fn test_broadcast() {
    let a1 = [1usize, 5, 1];
    let a2 = [4usize, 1, 1, 2];
    let h1: Ix3 = a1.into_dimension();
    let h2 = a2.into_dimension();

    let h3 = broadcast(&h1, &h2).unwrap();

    println!("{:?}", h3);
}

#[test]
fn test_binops() {
    let mut alloc = ArrayAllocator::<f32>::new();
    let a = alloc.alloc_randn([4, 4]);
    let b = alloc.alloc_randn([4, 4]);
    
    let c_id = alloc.binop(&a, &b, |a, b| a + b);
    let a0 = alloc.get_copy(&a).to_owned();
    let b0 = alloc.get_copy(&b).to_owned();
    let c = alloc.get_copy(&c_id);
    let c0 = &a0 + &b0;
    let c1 = &alloc.get_copy(&a) + &alloc.get_copy(&b);


    let err1 = (&c - &c0).fold(-f32::MAX, |a, acc| a.max((*acc).abs()));
    let err2 = (&c - &c1).fold(-f32::MAX, |a, acc| a.max((*acc).abs()));
    let err3 = (&c0 - &c1).fold(-f32::MAX, |a, acc| a.max((*acc).abs()));
    
    println!("max error {}", err1);
    println!("max error {}", err2);
    println!("max error {}", err3);
}

#[test]
fn test_operation() {
    let mut alloc = ArrayAllocator::<f32>::new();
    let a = alloc.alloc_randn([4, 4]);
    let b = alloc.alloc_randn([4, 4]);

    let mut c = alloc.alloc_uninit([4, 4]);
    alloc.x(&a).x(&b).xmut(&mut c).op(|((a, b), mut c)| {
        c += &(&a + &b);
    });
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