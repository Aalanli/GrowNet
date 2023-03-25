use super::allocator::{Alloc, AllocId};
pub use super::allocator::Partition;

use ndarray::{self as nd, ArrayView, ArrayViewMut, Dimension, IntoDimension};
use num_traits::Float;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform, Normal, Uniform};


pub struct ArrayAlloc {
    alloc: Alloc,
}

#[derive(Copy, Clone)]
pub struct ArrayId<T, D> {
    id: AllocId<T>,
    dim: D
}

pub struct ArrayAllocView<'a, T, D> {
    view: ArrayViewMut<'a, T, D>,
    part: Partition
}

impl<T, D: Dimension> ArrayId<T, D> {
    pub fn new(mut slice: ArrayViewMut<T, D>, partition: Partition) -> Option<Self> {
        let dim = slice.raw_dim();
        if let Some(xs) = slice.as_slice_mut() {
            Some(Self { id: AllocId::new(xs, partition), dim })
        } else {
            None
        }
    }

    pub fn dim(&self) -> D {
        self.dim.clone()
    }
}

impl<'a, T, D: Dimension> ArrayAllocView<'a, T, D> {
    pub fn view(&mut self) -> &mut ArrayViewMut<'a, T, D> {
        &mut self.view
    }

    pub fn id(self) -> ArrayId<T, D> {
        ArrayId::new(self.view, self.part).unwrap() // this is guaranteed to be contiguous
    }

    pub fn destructure(self) -> (ArrayViewMut<'a, T, D>, Partition) {
        (self.view, self.part)
    }
}

impl<'a, T: Clone, D: Dimension> ArrayAllocView<'a, T, D> {
    pub fn mapv(mut self, f: impl FnMut(T) -> T) -> Self {
        self.view.mapv_inplace(f);
        self
    }

    pub fn fill(self, v: T) -> Self {
        self.mapv(|_| v.clone())
    }
}

impl<'a, T: Float, D: Dimension> ArrayAllocView<'a, T, D> {
    pub fn zeros(self) -> Self {
        self.fill(T::zero())
    }

    pub fn ones(self) -> Self {
        self.fill(T::one())
    }
}

impl<'a, T: Float, D: Dimension> ArrayAllocView<'a, T, D>
where StandardNormal: Distribution<T> {
    pub fn randn(self) -> Self {
        let mut rng = thread_rng();
        let sampler = Normal::new(T::zero(), T::one()).unwrap();
        self.mapv(|_| sampler.sample(&mut rng))
    }    
}

impl<'a, T: Float, D: Dimension> ArrayAllocView<'a, T, D>
where T: SampleUniform {
    pub fn randu(self) -> Self {
        let mut rng = thread_rng();
        let sampler = Uniform::new(T::zero(), T::one());
        self.mapv(|_| sampler.sample(&mut rng))
    }    
}

impl ArrayAlloc {
    pub fn new() -> Self {
        Self { alloc: Alloc::new() }
    }

    pub fn request<T: Copy, D: Dimension, Sh: IntoDimension<Dim = D>>(&self, dim: Sh) -> ArrayAllocView<T, D> {
        let dim = dim.into_dimension();
        let (slice, part) = self.alloc.request_slice(dim.size()).destructure();
        let view = ArrayViewMut::from_shape(dim, slice).unwrap();
        ArrayAllocView { view, part }
    }

    pub fn to_slice<T, D: Dimension>(&self, id: ArrayId<T, D>) -> Option<ArrayView<T, D>> {
        if let Some(xs) = self.alloc.borrow_slice(id.id) {
            Some(ArrayView::from_shape(id.dim, xs).unwrap())
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.alloc.clear();
    }
}

