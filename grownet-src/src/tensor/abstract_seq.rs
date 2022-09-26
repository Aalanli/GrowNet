use std::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, RangeFull};
use std::slice::{Iter, IterMut};


pub trait AbsSeq
    : Index<usize,               Output =  <Self as AbsSeq>::ElemT> 
    + Index<Range<usize>,        Output = [<Self as AbsSeq>::ElemT]>
    + Index<RangeTo<usize>,      Output = [<Self as AbsSeq>::ElemT]>
    + Index<RangeFrom<usize>,    Output = [<Self as AbsSeq>::ElemT]>
    + Index<RangeFull,           Output = [<Self as AbsSeq>::ElemT]>
    + IndexMut<usize,            Output =  <Self as AbsSeq>::ElemT>
    + IndexMut<Range<usize>,     Output = [<Self as AbsSeq>::ElemT]>
    + IndexMut<RangeTo<usize>,   Output = [<Self as AbsSeq>::ElemT]>
    + IndexMut<RangeFrom<usize>, Output = [<Self as AbsSeq>::ElemT]>
    + IndexMut<RangeFull,        Output = [<Self as AbsSeq>::ElemT]>
{
    type ElemT;
    fn len(&self) -> usize;
    fn into_iter(&self) -> Iter<Self::ElemT> {
        self[..].iter()
    }
    fn into_mut_iter(&mut self) -> IterMut<Self::ElemT> {
        self[..].iter_mut()
    }
}

impl<T> AbsSeq for Vec<T> {
    type ElemT = T;
    fn len(&self) -> usize {
        Vec::<T>::len(self)
    }
}

impl<T, const N: usize> AbsSeq for [T; N] {
    type ElemT = T;
    fn len(&self) -> usize {
        N
    }
}

impl<T> AbsSeq for [T] {
    type ElemT = T;
    fn len(&self) -> usize {
        self.len()
    }
}