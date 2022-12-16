/// This module only defines the dataset logic for loading and processing datasets
/// This is separate from the data_ui module, which deals with integrating with the ui
/// for visualizations, etc.
/// 
/// Separating the logic can enable headlessmode which will be for future work

use ndarray::prelude::*;
use anyhow::Result;

pub mod data;
pub mod cifar;
pub mod mnist;

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset: Sync + Send {
    type DataPoint;
    fn next(&mut self) -> Option<Self::DataPoint>;
    fn reset(&mut self);
    fn shuffle(&mut self);
}

pub trait DatasetBuilder {
    type Dataset: Dataset;
    fn build_train(&self) -> Result<Self::Dataset>;
    fn build_test(&self) -> Option<Result<Self::Dataset>>;
}