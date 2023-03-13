// taken from https://github.com/srenevey/neuro with slight modifications
use arrayfire::*;
use super::utils;
use super::Float;

/// Used to generate the initial values for the parameters of the model.
#[derive(Debug, Copy, Clone)]
pub enum Initializer<T: Float> {
    /// Given constant value.
    Constant(T),
    /// Normal distribution scaled using Glorot scale factor.
    GlorotNormal,
    /// Uniform distribution scaled using Glorot scale factor.
    GlorotUniform,
    /// Normal distribution scaled using He scale factor.
    HeNormal,
    /// Uniform distribution scaled using He scale factor.
    HeUniform,
    /// Normal distribution scaled using Lecun scale factor.
    LecunNormal,
    /// Uniform distribution scaled using Lecun scale factor.
    LecunUniform,
    /// Normal distribution with mean 0 and standard deviation 0.01.
    Normal,
    /// Normal distribution with given mean and standard deviation.
    NormalScaled(T, T),
    /// Ones.
    Ones,
    /// Uniform distribution within -0.01 and 0.01.
    Uniform,
    /// Uniform distribution within the given bounds.
    UniformBounded(T, T),
    /// Zeros.
    Zeros,
}


impl<T: Float> Initializer<T> {

    /// Creates a tensor with random values generated from the distribution specified by the initializer.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions of the tensor created.
    /// * `fan_in` - The number of input units.
    /// * `fan_out` - The number of output units.
    pub(crate) fn init(self,
                             dims: Dim4,
                             fan_in: u64,
                             fan_out: u64
    ) -> Array<T> {
        match self {
            Initializer::Constant(x) => constant(x, dims),
            Initializer::GlorotNormal => {
                let standard_deviation = T::from(2. / (fan_out + fan_in) as f32).unwrap().sqrt();
                utils::scaled_normal(T::zero(), standard_deviation.into(), dims)
            },
            Initializer::GlorotUniform => {
                let limit = T::from(6. / (fan_out + fan_in) as f32).unwrap().sqrt();
                utils::scaled_uniform((-limit).into(), limit.into(), dims)
            },
            Initializer::HeNormal => {
                let standard_deviation = T::from(2. / fan_in as f32).unwrap().sqrt();
                utils::scaled_normal(T::zero(), standard_deviation.into(), dims)
            },
            Initializer::HeUniform => {
                let limit = T::from(6. / fan_in as f32).unwrap().sqrt();
                utils::scaled_uniform((-limit).into(), limit.into(), dims)
            },
            Initializer::LecunNormal => {
                let standard_deviation = T::from(1. / fan_in as f32).unwrap().sqrt();
                utils::scaled_normal(T::zero(), standard_deviation.into(), dims)
            },
            Initializer::LecunUniform => {
                let limit = T::from(3. / fan_in as f32).unwrap().sqrt();
                utils::scaled_uniform((-limit).into(), limit.into(), dims)
            },
            Initializer::Normal => utils::scaled_normal(T::zero(), T::from(0.01).unwrap().into(), dims),
            Initializer::NormalScaled(mean, standard_deviation) => utils::scaled_normal(mean, standard_deviation, dims),
            Initializer::Ones => utils::ones(dims),
            Initializer::Uniform => utils::scaled_uniform(T::from(-0.01).unwrap().into(), T::from(0.01).unwrap().into(), dims),
            Initializer::UniformBounded(lb, ub) => utils::scaled_uniform(lb, ub, dims),
            Initializer::Zeros => utils::zeros(dims),
        }
    }
}