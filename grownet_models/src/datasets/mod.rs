use std::marker::PhantomData;

use anyhow::Result;
/// This module only defines the dataset logic for loading and processing datasets
/// This is separate from the data_ui module, which deals with integrating with the ui
/// for visualizations, etc.
///
/// Separating the logic can enable headlessmode which will be for future work
use ndarray::prelude::*;

pub mod cifar;
pub mod data;
pub mod mnist;
pub mod transforms;

use super::Configure;

pub trait DatasetBuilder: Configure {
    type Dataset: Dataset;
    fn build_train(&self) -> Result<Self::Dataset>;
    fn build_test(&self) -> Option<Result<Self::Dataset>>;
}

/// The universal Dataset trait, which is the final object
/// passed to the model for training
pub trait Dataset {
    type DataPoint;
    fn next(&mut self) -> Option<Self::DataPoint>;
    fn reset(&mut self);
    fn shuffle(&mut self);
}

pub trait Transform<In, Out> {
    fn transform(&mut self, x: In) -> Out;
}

struct DataTransformer<D, T, Out> {
    dataset: D,
    transform: T,
    _out: PhantomData<Out>,
}

impl<In, Out, D, T> Dataset for DataTransformer<D, T, Out>
where
    D: Dataset<DataPoint = In>,
    T: Transform<In, Out>,
{
    type DataPoint = Out;

    fn next(&mut self) -> Option<Self::DataPoint> {
        let x = self.dataset.next();
        if let Some(x) = x {
            Some(self.transform.transform(x))
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.dataset.reset()
    }

    fn shuffle(&mut self) {
        self.dataset.shuffle()
    }
}

fn compose<In, Out, D, T>(d: D, t: T) -> impl Dataset<DataPoint = Out>
where
    D: Dataset<DataPoint = In>,
    T: Transform<In, Out>,
{
    DataTransformer {
        dataset: d,
        transform: t,
        _out: PhantomData,
    }
}
