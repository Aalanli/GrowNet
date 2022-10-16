#![allow(dead_code)]

mod m2;
mod tensor;
use std::{marker::PhantomData, ops::Index};

mod Test {
    use std::{marker::PhantomData, ops::Index};

    pub struct DynIndex<'a> {
        ptr: *const usize,
        len: usize,
        _marker: PhantomData<&'a usize>
    }

    pub struct LinArr<'a> {
        pub dims: *const usize,    // dynamic dimensions
        pub _marker: PhantomData<&'a usize>
    }

    pub trait ConvertIndex<'a> {
        type Result: 'a;
        fn convert(&self) -> Self::Result;
    }

    impl<'a> ConvertIndex<'a> for Vec<usize> {
        type Result = DynIndex<'a>;
        fn convert(&self) -> Self::Result {
            DynIndex{ptr: self.as_ptr(), len: self.len(), _marker: PhantomData}
        }
    }

    impl<'a, 'b> SIndex<LinArr<'a>> for DynIndex<'b> {
        fn tile_cartesian(&self, arr: &LinArr<'a>) -> Option<usize> {
            let i = unsafe {
                *self.ptr + *arr.dims
            };
            Some(i)
        }
    }

    struct TVec<T>(Vec<T>);

    impl<'a, T> TVec<T> {
        fn get_params<'t>(&self) -> LinArr<'t> {
            LinArr{ dims: &self.0.len() as *const usize, _marker: PhantomData}
        }
    }

    impl<'a, T, I> Index<I> for TVec<T> 
    where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: SIndex<LinArr<'a>> {
        type Output = T;
        fn index(&self, index: I) -> &Self::Output {
            let params = self.get_params(); 
            let i;
            {
                let ind = index.convert();
                i = ind.tile_cartesian(&params);

            }
            if let Some(j) = i {
                &self.0[j]
            } else {
                &self.0[1]
            }
        }
    }
    pub trait SIndex<Arr> {
        fn tile_cartesian(&self, arr: &Arr) -> Option<usize>;
    }
}

use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};
use std::ops::{IndexMut};
use std::iter::Iterator;
use std::fmt::Display;

pub struct WorldTensor<T> {
    pub ptr: NonNull<T>,         // NonNull pointer
    pub dims: Vec<usize>,        // dimensions row-major
    pub strides: Vec<usize>,     // strides row-major
    pub nelems: usize,              // total n elements
    pub marker: PhantomData<T>,  // tells compiler that Tensor owns values of type T
}
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![];
    strides.reserve_exact(dims.len());
    let mut k = 1usize;
    for i in (0..dims.len()).rev() {
        strides.push(k);
        k *= dims[i];
    }
    strides.reverse();
    strides
}

impl<T> WorldTensor<T> {
    pub fn new(dims: Vec<usize>) -> WorldTensor<T> {
        let strides= compute_strides(&dims);
        let nelems = strides[0] * dims[0];
        assert!(nelems > 0, "cannot make 0 sized tensor");

        let layout = Layout::array::<T>(nelems).unwrap();

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None  => alloc::handle_alloc_error(layout) 
        };

        WorldTensor{ptr, dims, strides, nelems, marker: PhantomData}
    }

    fn index_params(&self) -> LinArr {
        LinArr { dims: self.dims.as_ptr(), strides: self.strides.as_ptr(), len: self.dims.len(), _marker: PhantomData }
    }

}

pub struct LinArr<'a> {
    pub dims: *const usize,    // dynamic dimensions
    pub strides: *const usize, // dynamic strides
    pub len: usize,            // length of the dims and strides array
    pub _marker: PhantomData<&'a usize>
}

pub trait ConvertIndex<'a> {
    type Result: 'a;
    fn convert(&self) -> Self::Result;
}

impl<'a> ConvertIndex<'a> for Vec<usize> {
    type Result = DynIndex<'a>;
    fn convert(&self) -> Self::Result {
        DynIndex{ptr: self.as_ptr(), len: self.len(), _marker: PhantomData}
    }
}

pub struct DynIndex<'a> {
    ptr: *const usize,
    len: usize,
    _marker: PhantomData<&'a usize>
}

pub trait SIndex<Arr> {
    fn tile_cartesian(&self, arr: &Arr) -> Option<usize>;
}

impl<'a> SIndex<LinArr<'a>> for DynIndex<'a> {
    #[inline]
    fn tile_cartesian(&self, arr: &LinArr) -> Option<usize> {
        let offset = arr.len - self.len;
        let mut idx = 0;
        for i in 0..self.len {
            unsafe {
                let j = i + offset;
                let ind_i = *self.ptr.add(i);
                if ind_i >= *arr.dims.add(j) {
                    return None;
                }
                let stride = *arr.strides.add(j);
                idx += ind_i * stride;
            }
        }
        Some(idx)
    }
}

fn inbounds_get_ptr<T>(ptr: &NonNull<T>, i: usize) -> &T {
    unsafe{ ptr.as_ptr().add(i).as_ref().unwrap() }
}

impl<'a, T, I> Index<I> for WorldTensor<T> 
where I: ConvertIndex<'a>, <I as ConvertIndex<'a>>::Result: SIndex<LinArr<'a>>
{
    type Output = T;
    fn index(&self, ind: I) -> &T {
        let uindex = ind.convert();
        let index_params = LinArr { dims: self.dims.as_ptr(), strides: self.strides.as_ptr(), len: self.dims.len(), _marker: PhantomData };
        let i = uindex.tile_cartesian(&index_params);
        if let Some(idx) = i {
            // last check just to make sure
            if idx < self.nelems {
                return inbounds_get_ptr(&self.ptr, idx);
            }
        }
        panic!("Index for WorldTensor out of bounds");
    }
}


fn main() {
}
