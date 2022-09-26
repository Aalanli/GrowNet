use std::marker::PhantomData;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use num::{self, Float};

pub trait BackProp {
    type Args;
    type Results;
    type Grads;
    fn forward(args: Self::Args) -> Self::Results;
    fn backward(args: Self::Results) -> Self::Grads;
}

pub trait ComputeEdge {
    type fType;
}

struct SimpleLinear<T: num::Float> {
    w: Array2<T>,
    b: Array1<T>
}

//impl<T: num::Float> BackProp for SimpleLinear<T> {
//    type Args = Array1<T>;
//    type Grads = (Array1<T>);
//}

struct ComputeNode<M: ComputeEdge> {
    model: M,
    
}