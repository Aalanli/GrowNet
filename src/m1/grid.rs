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
    pub fn new(params: GlobalParams) -> Self {
        let grid = WorldTensor::new(params.grid_dim.clone(), || ComputeInstance::new(params.compute_dim));
        let indexing = I::new(&params.grid_dim);
        Baseline { indexing, grid, params }
    }

    //pub fn forward<I>(inputs: &I) -> 
}