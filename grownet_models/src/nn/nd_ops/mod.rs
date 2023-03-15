mod utils;
pub mod context;
pub mod ops_owned;
pub mod ops_ctx;

pub use utils::*;
pub use ops_ctx as ctx;
pub use ops_owned as owned;

use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform, Normal, Uniform};
use ndarray as nd;
use nd::prelude::*;
use nd::{RemoveAxis, IntoDimension, Zip};
use num::Float;
use num_traits::FromPrimitive;

use ndarray_rand::{rand::thread_rng, RandomExt};


pub struct Param<T, D> {
    pub w: Array<T, D>,
    pub g: Array<T, D>,
}

impl<T: Float, D: Dimension> Param<T, D>
where StandardNormal: Distribution<T> 
{
    pub fn zeros<Sh: IntoDimension<Dim = D> + Clone>(dim: Sh) -> Param<T, D> {
        Param { w: Array::zeros(dim.clone()), g: Array::zeros(dim) }
    }

    pub fn randn<Sh: IntoDimension<Dim = D> + Clone>(dim: Sh) -> Param<T, D> {
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



