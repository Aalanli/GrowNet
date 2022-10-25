use crate::tensor::WorldTensor;
use super::compute::ComputeInstance;
use super::indexing::{self as ind, IndexPolicy};
use super::GlobalParams;

/// Baseline has no sparsity metric, so it propagates its signals throughout the whole
/// grid, very similar to a convolutional neural network
pub struct Baseline<I> {
    indexing: I,
    grid: WorldTensor<ComputeInstance>,
    params: GlobalParams
}

impl<I> Baseline<I> 
where I: for<'a> IndexPolicy<'a> 
{
    pub fn new(dims: &[usize], params: GlobalParams) -> Self {
        let grid = WorldTensor::new(dims.to_vec(), || ComputeInstance::new(params.compute_dim));
        let indexing = I::new(dims);
        Baseline { indexing, grid, params }
    }
}