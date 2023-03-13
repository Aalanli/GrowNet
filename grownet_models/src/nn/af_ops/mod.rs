use half::f16;

use af::{ConstGenerator, FloatingPoint, RealFloating, Convertable, Dim4};
pub use arrayfire::{self as af, Array, HasAfEnum, dim4};

use crate::Flatten;

pub mod initializer;
pub mod conv;
pub mod linear;
pub mod maxpool;
pub mod activations;
pub mod batchnorm2d;
pub mod instancenorm;
pub mod utils;
pub mod array_ops;
pub mod loss;

use initializer as init;

pub use utils::{ones, zeros};
pub use array_ops::{reshape, reduce_sum};


pub struct Param<T: Float> {
    pub w: Array<T>,
    pub g: Array<T>
}

impl<T: Float> Param<T> {
    pub fn new(w: Array<T>) -> Param<T> {
        let g = af::constant(T::zero(), w.dims());
        Param { w, g }
    }

    pub fn dims(&self) -> Dim4 {
        self.w.dims()
    }
}

impl<T: Float + 'static> Flatten for Param<T> {
    fn flatten<'a>(&'a mut self, path: String, world: &mut crate::World<'a>) {
        world.push(path, self);
    }
}

pub trait Float: 
    num::Float + 
    HasAfEnum + 
    FloatingPoint + 
    ConstGenerator <
        OutType = Self, 
        AbsOutType = Self, 
        AggregateOutType = Self, 
        InType = Self,
        UnaryOutType = Self,
        MeanOutType = Self> + 
    Copy + 
    RealFloating + 
    Convertable<OutType = Self>
    + 'static {}

impl Float for f32 {}
impl Float for f64 {}