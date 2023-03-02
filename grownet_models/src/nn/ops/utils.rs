use arrayfire::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::Float;

pub fn scaled_uniform<T: Float>(lower_bound: T, upper_bound: T, dims: Dim4) -> Array<T> {
    constant(lower_bound, dims) + constant(upper_bound.sub(lower_bound), dims) * randu::<T>(dims)
}

pub fn scaled_normal<T: Float>(mean: T, standard_deviation: T, dims: Dim4) -> Array<T> {
    constant(standard_deviation, dims) * randn::<T>(dims) + constant(mean, dims)
}

pub fn ones<T: Float>(dims: Dim4) -> Array<T> {
    constant(T::one(), dims)
}

pub fn zeros<T: Float>(dims: Dim4) -> Array<T> {
    constant(T::zero(), dims)
}
