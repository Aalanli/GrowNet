use half::f16;

use af::{ConstGenerator, FloatingPoint, RealFloating, Convertable};
use arrayfire::{self as af, Array, HasAfEnum, dim4};

pub mod initializer;
pub mod conv;
pub mod linear;
pub mod maxpool;
pub mod activations;
pub mod batchnorm2d;
pub mod instancenorm2d;
pub mod utils;

use initializer as init;

pub type RcArray<T> = Array<T>;

pub struct Param<T: Float> {
    w: Array<T>,
    g: Array<T>
}

impl<T: Float> Param<T> {
    fn new(w: Array<T>) -> Param<T> {
        let g = af::constant(T::zero(), w.dims());
        Param { w, g }
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
    Convertable<OutType = Self> {}

impl Float for f32 {}
impl Float for f64 {}