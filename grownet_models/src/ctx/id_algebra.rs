use std::ops::{Add, Sub, Mul, Div, Neg};

use ndarray::{self as nd, ArrayView, ArrayViewMut, Dimension, IntoDimension, DimMax, NdProducer};
use anyhow::{Error, Result, Context};
use num_traits::{Float, Num};

use super::*;


type ExecOut<'a, T, D> = ArrayAllocView<'a, T, D>;
pub trait Exec {
    type F;
    type OutDim: Dimension;
    fn exec(self, ctx: &ArrayAlloc) -> ExecOut<Self::F, Self::OutDim>;
    fn outdim(&self) -> Self::OutDim;
}

pub trait BroadCastIter<T, D> {
    type Iter<'a>: Iterator<Item = T> where T: 'a;
    fn broadcast<'a>(self, dim: D, ctx: &'a ArrayAlloc) -> Self::Iter<'a>;
}

impl<T: Copy, D: Dimension> Exec for ArrayId<T, D> {
    type F = T;
    type OutDim = D;
    #[inline]
    fn exec(self, ctx: &ArrayAlloc) -> ExecOut<Self::F, Self::OutDim> {
        ctx.request(self.dim())
    }
    #[inline]
    fn outdim(&self) -> Self::OutDim {
        self.dim()
    }
}

pub struct OwnedIter<It>(It);
impl<'a, T: 'a + Clone, It> Iterator for OwnedIter<It>
where It: Iterator<Item = &'a T> 
{
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| { x.clone() })
    }
}

impl<T: Clone, D1: Dimension, D2> BroadCastIter<T, D2> for ArrayId<T, D1>
where D2: DimMax<D1> + Dimension + PartialEq<D1>
{
    type Iter<'a> = OwnedIter<nd::iter::Iter<'a, T, D2>> where T: 'a;
    #[inline]
    fn broadcast<'a>(self, dim: D2, ctx: &'a ArrayAlloc) -> Self::Iter<'a> {
        let original_dim = self.dim();
        let view = ctx.to_slice(self).expect("unable to retrieve view");
        let arr = if dim == original_dim {
            unsafe { 
                let ptr = view.as_ptr();
                ArrayView::from_shape_ptr(dim, ptr)
            }
        } else {
            let arr: ArrayView<T, D2> = view.broadcast(dim.clone()).expect("unable to broadcast");
            let arr = unsafe { 
                std::mem::transmute::<ArrayView<T, D2>, ArrayView<'a, T, D2>>(arr)
            };
            arr
        };
        OwnedIter(arr.into_iter())
    }
}

impl<T: Num + Copy + 'static, D2: Dimension> BroadCastIter<T, D2> for T {
    type Iter<'a> = std::iter::Repeat<T> where T: 'a;

    fn broadcast<'a>(self, _dim: D2, _ctx: &'a ArrayAlloc) -> Self::Iter<'a> {
        std::iter::repeat(self)
    }
}

impl<T: Num + Copy> Exec for T {
    type F = T;
    type OutDim = Ix1;

    fn exec(self, ctx: &ArrayAlloc) -> ExecOut<Self::F, Self::OutDim> {
        let mut alloc = ctx.request::<Self::F, _, _>((1,));
        alloc.view().fill(self);
        alloc
    }

    fn outdim(&self) -> Self::OutDim {
        (1,).into_dimension()
    }
}

pub struct Scalar<T>(pub T);

pub trait BinFn<A, B=A> {
    type C;
    fn apply(a: A, b: B) -> Self::C;
}

pub struct UniExec<A, F> {
    arr: A,
    f: F
}

pub struct BinExec<A, B, F> {
    lhs: A,
    rhs: B,
    f: F
}

pub struct UniExecIter<It, F>(It, F);

pub struct BinExecIter<It1, It2, F>(It1, It2, F);

macro_rules! derive_opf {
    ($opF:ident, $fn:tt) => {
        pub struct $opF;

        impl<A: Num> BinFn<A> for $opF {
            type C = A;
            #[inline]
            fn apply(a: A, b: A) -> Self::C {
                a $fn b
            }
        }    
    };

    ($opF:ident, $type_out:ty, $fn:tt) => {
        pub struct $opF;

        impl<A: Num> BinFn<A> for $opF {
            type C = $type_out;
            #[inline]
            fn apply(a: A, b: A) -> Self::C {
                a $fn b
            }
        }    
    };
}

derive_opf!(AddF, +);
derive_opf!(SubF, -);
derive_opf!(MulF, *);
derive_opf!(DivF, /);
derive_opf!(EqF, bool, ==);

macro_rules! derive_ops_expr {
    ($op:ident, $opf:ident, $fn_name:ident) => {
        impl<T: Exec, A, B, F> $op<T> for BinExec<A, B, F> {
            type Output = BinExec<BinExec<A, B, F>, T, $opf>;
            #[inline]
            fn $fn_name(self, rhs: T) -> Self::Output {
                BinExec { lhs: self, rhs, f: $opf }
            }
        }
        
        impl<T: Num + Clone, A, B, F> $op<BinExec<A, B, F>> for Scalar<T> {
            type Output = BinExec<T, BinExec<A, B, F>, $opf>;
            #[inline]
            fn $fn_name(self, rhs: BinExec<A, B, F>) -> Self::Output {
                BinExec { lhs: self.0, rhs, f: $opf }
            }
        }
        
        impl<T1, T2, D: Dimension> $op<T2> for ArrayId<T1, D> {
            type Output = BinExec<ArrayId<T1, D>, T2, $opf>;
            #[inline]
            fn $fn_name(self, rhs: T2) -> Self::Output {
                BinExec { lhs: self, rhs, f: $opf }
            }
        }
        
        impl<T: Num + Clone, D: Dimension> $op<ArrayId<T, D>> for Scalar<T> {
            type Output = BinExec<T, ArrayId<T, D>, $opf>;
            #[inline]
            fn $fn_name(self, rhs: ArrayId<T, D>) -> Self::Output {
                BinExec { lhs: self.0, rhs, f: $opf }
            }
        }
    };
}

derive_ops_expr!(Add, AddF, add);
derive_ops_expr!(Sub, SubF, sub);
derive_ops_expr!(Mul, MulF, mul);
derive_ops_expr!(Div, DivF, div);

macro_rules! derive_unit_ops {
    ($op:ident(self) -> Self) => {
        pub fn $op<F: Float, A>(a: A) -> UniExec<A, impl Fn(F) -> F> {
            UniExec { arr: a, f: |a: F| a.$op() }
        }
    };

    ($op:ident(self) -> $out:tt) => {
        pub fn $op<F: Float, A>(a: A) -> UniExec<A, impl Fn(F) -> $out> {
            UniExec { arr: a, f: |a: F| a.$op() }
        }
    };

    ($op:ident(self, $var:ident: Self) -> Self) => {
        pub fn $op<F: Float, A>(a: A, $var: F) -> UniExec<A, impl Fn(F) -> F> {
            UniExec { arr: a, f: move |a: F| a.$op($var) }
        }
    };

    ($op:ident(self, $var:ident: $other:tt) -> Self) => {
        pub fn $op<F: Float, A>(a: A, $var: $other) -> UniExec<A, impl Fn(F) -> F> {
            UniExec { arr: a, f: move |a: F| a.$op($var) }
        }
    };

    ($op:ident(self, $var:ident: Self) -> $out:tt) => {
        pub fn $op<F: Float, A>(a: A, $var: F) -> UniExec<A, impl Fn(F) -> $out> {
            UniExec { arr: a, f: |a: F| a.$op($var) }
        }
    };

    ($op:ident(self, $var:ident: $other:tt) -> $out:tt) => {
        pub fn $op<F: Float, A>(a: A, $var: $other) -> UniExec<A, impl Fn(F) -> $out> {
            UniExec { arr: a, f: |a: F| a.$op($var) }
        }
    };
}

derive_unit_ops!(is_nan(self) -> bool);
derive_unit_ops!(is_infinite(self) -> bool);
derive_unit_ops!(is_finite(self) -> bool);
derive_unit_ops!(is_normal(self) -> bool);
derive_unit_ops!(floor(self) -> Self);
derive_unit_ops!(ceil(self) -> Self);
derive_unit_ops!(round(self) -> Self);
derive_unit_ops!(trunc(self) -> Self);
derive_unit_ops!(fract(self) -> Self);
derive_unit_ops!(abs(self) -> Self);
derive_unit_ops!(signum(self) -> Self);
derive_unit_ops!(is_sign_positive(self) -> bool);
derive_unit_ops!(is_sign_negative(self) -> bool);
derive_unit_ops!(recip(self) -> Self);
derive_unit_ops!(powi(self, n: i32) -> Self);
derive_unit_ops!(powf(self, n: Self) -> Self);
derive_unit_ops!(sqrt(self) -> Self);
derive_unit_ops!(exp(self) -> Self);
derive_unit_ops!(exp2(self) -> Self);
derive_unit_ops!(ln(self) -> Self);
derive_unit_ops!(log(self, base: Self) -> Self);
derive_unit_ops!(log2(self) -> Self);
derive_unit_ops!(log10(self) -> Self);
derive_unit_ops!(to_degrees(self) -> Self);
derive_unit_ops!(to_radians(self) -> Self);
derive_unit_ops!(max(self, other: Self) -> Self);
derive_unit_ops!(min(self, other: Self) -> Self);
derive_unit_ops!(cbrt(self) -> Self);
derive_unit_ops!(hypot(self, other: Self) -> Self);
derive_unit_ops!(sin(self) -> Self);
derive_unit_ops!(cos(self) -> Self);
derive_unit_ops!(tan(self) -> Self);
derive_unit_ops!(asin(self) -> Self);
derive_unit_ops!(acos(self) -> Self);
derive_unit_ops!(atan(self) -> Self);
derive_unit_ops!(atan2(self, other: Self) -> Self);
derive_unit_ops!(exp_m1(self) -> Self);
derive_unit_ops!(ln_1p(self) -> Self);
derive_unit_ops!(sinh(self) -> Self);
derive_unit_ops!(cosh(self) -> Self);
derive_unit_ops!(tanh(self) -> Self);
derive_unit_ops!(asinh(self) -> Self);
derive_unit_ops!(acosh(self) -> Self);
derive_unit_ops!(atanh(self) -> Self);

impl<'a, X, Y, F, It> Iterator for UniExecIter<It, F>
where It: Iterator<Item = X>, X: 'a, F: Fn(X) -> Y
{
    type Item = Y;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.0.next() {
            Some(self.1(x))
        } else {
            None
        }
    }
}

impl<'a, TA, TB, TC, It1, It2, F> Iterator for BinExecIter<It1, It2, F>
where It1: Iterator<Item = TA>, It2: Iterator<Item = TB>, TA: 'a, TB: 'a, TC: 'a, F: BinFn<TA, TB, C=TC>
{
    type Item = TC;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let a = self.0.next();
        let b = self.1.next();
        if a.is_some() && b.is_some() {
            let a = a.unwrap();
            let b = b.unwrap();
            Some(F::apply(a, b))
        } else {
            None
        }
    }
}


impl<X, Y, A, D, F> Exec for UniExec<A, F>
where
    A: Exec<F = X, OutDim = D> + BroadCastIter<X, D>,
    D: Dimension,
    F: Fn(X) -> Y,
    Y: Copy
{
    type F = Y;
    type OutDim = D;
    #[inline]
    fn exec(self, ctx: &ArrayAlloc) -> ExecOut<Self::F, Self::OutDim> {
        let outdim = self.outdim();
        let exec = UniExecIter(self.arr.broadcast(outdim.clone(), ctx), self.f);
        let mut alloc_view = ctx.request::<Y, _, _>(outdim);
        alloc_view.view().iter_mut().zip(exec).for_each(|(x, y)| {
            *x = y;
        });
        alloc_view
    }

    #[inline]
    fn outdim(&self) -> Self::OutDim {
        self.arr.outdim()
    }
}

impl<TA, TB, TC, A, B, D1, D2, F> Exec for BinExec<A, B, F> 
where
    A: Exec<F = TA, OutDim = D1> + BroadCastIter<TA, <D2 as DimMax<D1>>::Output>, 
    B: Exec<F = TB, OutDim = D2> + BroadCastIter<TB, <D2 as DimMax<D1>>::Output>,
    D1: Dimension,
    D2: Dimension + DimMax<D1>,
    F: BinFn<TA, TB, C = TC>,
    TC: Copy,
{
    type F = TC;
    type OutDim = <D2 as DimMax<D1>>::Output;
    #[inline]
    fn exec(self, ctx: &ArrayAlloc) -> ExecOut<Self::F, Self::OutDim> {
        let outdim = self.outdim();
        let mut alloc_view = ctx.request::<TC, _, _>(outdim.clone());
        let it1 = self.lhs.broadcast(outdim.clone(), ctx);
        let it2 = self.rhs.broadcast(outdim, ctx);
        let addexec = BinExecIter(it1, it2, self.f);
        alloc_view.view().iter_mut().zip(addexec).for_each(|(x, y)| {
            *x = y;
        });
        alloc_view
    }
    
    #[inline]
    fn outdim(&self) -> Self::OutDim {
        let dim1 = self.lhs.outdim();
        let dim2 = self.rhs.outdim();
        match broadcast(&dim1, &dim2) {
            Ok(xs) => xs,
            Err(_) => panic!("unable to broadcast {:?} + {:?}", dim1, dim2)
        }
    }
}

impl<X, Y, A, F, D> BroadCastIter<Y, D> for UniExec<A, F>
where
    X: 'static,
    Y: 'static,
    A: Exec<F = X, OutDim = D> + BroadCastIter<X, D>,
    F: Fn(X) -> Y,
    D: Dimension
{
    type Iter<'a> = UniExecIter<<A as BroadCastIter<X, D>>::Iter<'a>, F>;
    
    #[inline]
    fn broadcast<'a>(self, dim: D, ctx: &'a ArrayAlloc) -> Self::Iter<'a> {
        let it = self.arr.broadcast(dim, ctx);
        UniExecIter(it, self.f)
    }
}

impl<TA, TB, TC, A, B, F, D1, D2> BroadCastIter<TC, <D2 as DimMax<D1>>::Output> for BinExec<A, B, F> 
where
    TA: 'static,
    TB: 'static,
    TC: Copy + 'static,
    F: BinFn<TA, TB, C = TC>,
    A: Exec<F = TA, OutDim = D1> + BroadCastIter<TA, <D2 as DimMax<D1>>::Output>, 
    B: Exec<F = TB, OutDim = D2> + BroadCastIter<TB, <D2 as DimMax<D1>>::Output>,
    D1: Dimension,
    D2: Dimension + DimMax<D1> 
{
    type Iter<'s> = BinExecIter<<A as BroadCastIter<TA, <D2 as DimMax<D1>>::Output>>::Iter<'s>, 
        <B as BroadCastIter<TB, <D2 as DimMax<D1>>::Output>>::Iter<'s>, F>;

    #[inline]
    fn broadcast<'a>(self, dim: <D2 as DimMax<D1>>::Output, ctx: &'a ArrayAlloc) -> Self::Iter<'a> {
        let it1 = self.lhs.broadcast(dim.clone(), ctx);
        let it2 = self.rhs.broadcast(dim, ctx);
        BinExecIter(it1, it2, self.f)
    }
}


use nd::{ShapeError, ErrorKind, Ix1};

/// broadcasts two dimensions for binary operations
#[inline]
pub fn broadcast<D1, D2>(shape1: &D1, shape2: &D2) -> Result<<D2 as DimMax<D1>>::Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension + DimMax<D1>,
    //<D2 as DimMax<D1>>::Output: Into<StrideShape<<D2 as DimMax<D1>>::Output>>
{
    co_broadcast::<D1, D2, <D2 as DimMax<D1>>::Output>(shape1, shape2)
}

/// Helper broadcasting function copied from ndarray source
#[inline]
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (_k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }

    let mut out = Output::default();
    for (out, s) in out.slice_mut().iter_mut().zip(shape1.slice().iter()) {
        *out = *s;
    }
    let h = out.slice_mut();
    let it = IntoIterator::into_iter(h);
    for (out, s2) in it.zip(shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError::from_kind(k)
}

#[test]
fn test_add() {
    let a = ArrayAlloc::new();
    let id1 = a.request::<f32, _, _>((1, 2)).randn().id();
    let id2 = a.request::<f32, _, _>((2, 2)).randn().id();
    let id3 = a.request::<f32, _, _>((2, 2)).randn().id();
    let c = powi(sin((id1 + id2 + id3) * id3 / id2), 2);
    let d = c.exec(&a);

    let view1 = a.to_slice(id1).unwrap();
    let view2 = a.to_slice(id2).unwrap();
    let view3 = a.to_slice(id3).unwrap();
    
    let (view4, _) = d.destructure();

    println!("{}", view1);
    println!("{:?}", view4.dim());
    let out = ((&view1 + &view2 + &view3) * &view3 / &view2).mapv(|x| x.sin().powi(2));
    println!("{}", out);
    println!("{}", view4);
}

#[test]
fn test_add1() {
    let alloc = ArrayAlloc::new();
    let id1 = alloc.request::<f32, _, _>((1, 2)).fill(1.0).id();
    let id2 = alloc.request::<f32, _, _>((2, 2)).fill(-1.0).id();

    println!("{:?}", (id1 + id2).outdim());
    let c = (id1 + id2).exec(&alloc);
    let (view, _) = c.destructure();

    let view1 = alloc.to_slice(id1).unwrap();
    let view2 = alloc.to_slice(id2).unwrap();
    let out1 = view1.broadcast((2, 2)).unwrap();
    let mut out2 = nd::Array2::<f32>::zeros((2, 2));
    out2.iter_mut().zip(out1.iter()).zip(view2.iter()).for_each(|((a, b), c)| {
        *a = *b + *c;
    });
    println!("{}", out2);

    let out = &view1 + &view2;

    println!("{}", view);
    println!("{}", out);
}
