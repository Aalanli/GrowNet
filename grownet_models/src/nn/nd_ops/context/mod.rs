use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray::{self as nd, RemoveAxis, IntoDimension};

use num::{Float, FromPrimitive};

mod flat;
mod block;
mod naive;

pub use flat::FlatCtx;
pub use block::BlockCtx;
pub use naive::NaiveCtx;

pub trait ArrayCtx<T: Float> {
    fn empty<'a, D: Dimension, Sh: IntoDimension<Dim = D> + Clone>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D>;
    
    fn clone<'a, D: Dimension>(&'a self, xs: &ArrayView<T, D>) -> ArrayViewMut<'a, T, D> {
        let mut empty = self.empty(xs.raw_dim());
        empty.assign(&xs);
        empty
    }

    fn zeros<'a, D: Dimension, Sh: IntoDimension<Dim = D> + Clone>(&'a self, dim: Sh) -> ArrayViewMut<'a, T, D> {
        let mut arr = self.empty(dim);
        arr.fill(T::zero());
        arr
    }

    fn clear(&mut self);
    
    /// This function only works if xs is a view of self, and panics otherwise.
    /// If copying a block of memory into the context is necessary, use 'clone'
    fn id<D: Dimension>(&self, xs: ArrayViewMut<T, D>) -> ArrId<T, D>;
    
    /// Since id only permits a mutable view from self to construct an ArrId, and the only
    /// way to construct a mutable view from self is through the ArrayCtx functions, ArrId is guaranteed 
    /// to be an unique representation of 'owned memory', referencing self; ArrId is non-clone and behaves as if
    /// it owns a chunk of memory. 
    fn from_id<D: Dimension>(&self, id: &ArrId<T, D>) -> ArrayView<T, D>;

    /// Therefore, it is safe to consume id and return a mutable view, since there is no aliasing going on.
    fn from_id_mut<D: Dimension>(&self, id: ArrId<T, D>) -> ArrayViewMut<T, D>;
}

/// Guaranteed to be unique for each view into ctx
pub struct ArrId<T, D> {
    dim: D,
    offset: usize,
    gen: usize,
    _data: PhantomData<T>
}
