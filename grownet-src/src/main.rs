#![allow(dead_code)]

use std::fmt::Display;
use std::fmt::format;
use std::ops::{Deref, Range, RangeFrom, RangeTo, RangeFull};


use num;


mod m2;
mod tensor;

macro_rules! test {
    ($($e1:pat),*) => {}
}

pub enum IndexInfo {
    Edge,
    Index(usize)
}

pub use IndexInfo::Edge as Edge;
pub use IndexInfo::Index as Index;


pub struct TsSlice((IndexInfo, IndexInfo));

impl Deref for TsSlice {
    type Target = (IndexInfo, IndexInfo);
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct StaticTsSlices<const N: usize>([TsSlice; N]);

impl<const N: usize> Deref for StaticTsSlices<N> {
    type Target = [TsSlice; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> StaticTsSlices<N> {
    fn len() -> usize {
        N
    }
}


impl Display for TsSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let l = if let Index(i) = self.0.0 {format!("{}", i)} else {String::from("")};
        let r = if let Index(i) = self.0.1 {format!("{}", i)} else {String::from("")};
        write!(f, "{}:{}", l, r)
    }
}

impl<const N: usize> Display for StaticTsSlices<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[").unwrap();
        for i in 0..N-1 {
            write!(f, "{}, ", self[i]).unwrap();
        }
        write!(f, "{}]", self[N-1])
    }
}

macro_rules! extract {
    (;)                 => {TsSlice((Edge,Edge))};
    (;$r:expr)          => {TsSlice((Edge, Index($r)))};
    ($l:expr;)          => {TsSlice((Index($l), Edge))};
    ($l:expr ; $r:expr) => {TsSlice((Index($l), Index($r)))};
}

trait SliceSpec {
    type Output;
    fn to_slice(&self) -> Self::Output;
}

impl SliceSpec for Range<usize> {
    type Output = TsSlice;
    fn to_slice(&self) -> Self::Output {
        TsSlice((Index(self.start), Index(self.end)))
    }
}

impl SliceSpec for RangeTo<usize> {
    type Output = TsSlice;
    fn to_slice(&self) -> Self::Output {
        TsSlice((Edge, Index(self.end)))
    }
}

impl SliceSpec for RangeFrom<usize> {
    type Output = TsSlice;
    fn to_slice(&self) -> Self::Output {
        TsSlice((Index(self.start), Edge))
    }
}

impl SliceSpec for RangeFull {
    type Output = TsSlice;
    fn to_slice(&self) -> Self::Output {
        TsSlice((Edge, Edge))
    }
}

macro_rules! count_exprs {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),*) => (1 + count_exprs!($($tail),*));
}

macro_rules! slice {
    ($($r:expr),*) => {
        {
            use std::mem::{self, MaybeUninit};

            const N: usize = count_exprs!($($r),*);
            let mut temp_data: [MaybeUninit<TsSlice>; N] = unsafe {
                MaybeUninit::uninit().assume_init()
            };
            let mut i: usize = 0;
            $(
                temp_data[i] = MaybeUninit::new(($r).to_slice());
                i += 1;
            )*

            let temp_data = unsafe {mem::transmute::<_, [TsSlice; N]>(temp_data)};
            StaticTsSlices::<N>(temp_data)
        }
    };
}

fn main() {
    let c = TsSlice((Edge, Edge));
    let i = 1;
    let s = slice![1..2, 2.., ..];
    println!("{}", s.len());
}
