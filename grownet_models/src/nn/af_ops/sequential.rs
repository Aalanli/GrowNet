use super::*;

use std::{rc::Rc, marker::PhantomData};

type DF<M, F> = Rc<dyn Fn(&mut M, &Array<F>) -> Array<F>>;

pub trait Seq<F: Float>: 'static {
    fn forward(&self, x: &Array<F>) -> (Array<F>, DF<Self, F>);
}

pub struct Sequential<H, T, F> {
    head: H,
    tail: T,
    _float: PhantomData<F>
}

impl<F: Float, H: Seq<F>, T: Seq<F>> Seq<F> for Sequential<H, T, F> {
    fn forward(&self, x: &Array<F>) -> (Array<F>, DF<Self, F>) {
        let (x1, df1) = self.head.forward(x);
        let (x2, df2) = self.tail.forward(&x1);

        let df = move |s: &mut Self, g: &Array<F>| {
            let df2 = df2.as_ref();
            let dx1 = df2(&mut s.tail, g);
            let df1 = df1.as_ref();
            let dx = df1(&mut s.head, &dx1);
            dx
        };

        (x2, Rc::new(df))
    }
}


impl<F: Float, H: Seq<F>, T: Seq<F>> Sequential<H, T, F> {
    pub fn new(h: H, t: T) -> Self {
        Self { head: h, tail: t, _float: PhantomData }
    }

    pub fn add<T1: Seq<F>>(self, new: T1) -> Sequential<impl Seq<F>, T1, F> {
        Sequential { head: self, tail: new, _float: PhantomData }
    }
}

pub struct Residual<A> {
    a: A,
}

impl<F: Float, A: Seq<F>> Seq<F> for Residual<A> {
    fn forward(&self, x: &Array<F>) -> (Array<F>, DF<Self, F>) {
        let (x1, dfx) = self.a.forward(x);
        let y = x1 + x; 
        let df = move |s: &mut Self, g: &Array<F>| {
            let dfx = dfx.as_ref();
            let g = g + dfx(&mut s.a, g);
            g
        };

        (y, Rc::new(df))
    }
}