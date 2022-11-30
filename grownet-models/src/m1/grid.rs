use ndarray::prelude as np;

use super::super::tensor::WorldTensor;
use super::compute::Linear;
use super::indexing::{self as ind, IndexPolicy};
use super::GlobalParams;

/// Baseline has no sparsity metric, so it propagates its signals throughout the whole
/// grid, very similar to a convolutional neural network
pub struct Baseline<I> {
    indexing: I,
    grid: WorldTensor<Linear>,
    params: GlobalParams
}

impl<I> Baseline<I> 
where I: for<'a> IndexPolicy<'a> 
{
    pub fn new(params: GlobalParams) -> Self {
        let grid = WorldTensor::new(params.grid_dim.clone(), || Linear::new(params.compute_dim));
        let indexing = I::new(&params.grid_dim);
        Baseline { indexing, grid, params }
    }

    // pub fn forward(inputs: &std::slice::Iter<(usize, np::Array1<f32>)>) -> &[f32] {
    //     
    // }
}

#[cfg(test)]
mod test {
    use ndarray as np;
    use std::{mem, any::Any};
    

    struct Test {
        a: Vec<usize>
    }
    #[test]
    fn size_test() {
        println!("{}", mem::needs_drop::<[f32; 3]>());
        println!("{}", mem::needs_drop::<f32>());
        println!("{}", mem::needs_drop::<Test>());
    }
}