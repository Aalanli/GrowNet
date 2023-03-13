mod utils;
mod instancenorm;

use rand_distr::{Distribution, StandardNormal};
pub use utils::*;
pub use instancenorm::*;

use ndarray as nd;
use nd::prelude::*;
use num::Float;
use ndarray_rand::{rand, rand_distr::Normal, RandomExt};

pub trait GDim: nd::ShapeBuilder + Clone {}
impl<T: nd::ShapeBuilder + Clone> GDim for T {}

pub struct Param<T, D> {
    pub w: Array<T, D>,
    pub g: Array<T, D>,
}

impl<T: Float, D: Dimension> Param<T, D>
where StandardNormal: Distribution<T> 
{
    pub fn zeros<Sh: GDim<Dim = D>>(dim: Sh) -> Param<T, D> {
        Param { w: Array::zeros(dim.clone()), g: Array::zeros(dim) }
    }

    pub fn randn<Sh: GDim<Dim = D>>(dim: Sh) -> Param<T, D> {
        let w = Array::random(dim.clone(), Normal::new(T::zero(), T::one()).unwrap());
        let g = Array::zeros(dim);
        Param { w, g }
    }
}

pub struct AdamParam<T, D> {
    param: Param<T, D>,
    mt: Array<T, D>,
    vt: Array<T, D>,
}



