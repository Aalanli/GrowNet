#![allow(dead_code)]
#[macro_use]
extern crate bencher;

use std::{alloc};

use bencher::Bencher;

use nalgebra::{self as na};
use ndarray::{self as nd};

const SHAPE: (usize, usize) = (128, 128);

fn test() {
    let m4 = na::Matrix2x3::from_column_slice(&[
        1.1, 2.1,
        1.2, 2.2,
        1.3, 2.3
    ]);
    let mut a: Vec<f32> = (0..18).map(|x| x as f32).collect();
    let c: na::Const<3>;
    let s = na::VecStorage::new(na::Const::<3>, na::Dynamic::new(6), a);
    unsafe {na::Matrix3xX::from_data_statically_unchecked(s);}

}

fn bench_nd(bench: &mut Bencher) {
    let shape = SHAPE;
    let nelems = shape.0 * shape.1;
    let a: Vec<f32>  = (0..nelems).map(|x| x as f32).collect();
    let b = a.clone();
    let mut c = a.clone();


    let A: nd::ArrayBase<_, nd::Ix2> = unsafe {nd::ArrayView::from_shape_ptr(shape, a.as_ptr())};
    let B: nd::ArrayBase<_, nd::Ix2>  = unsafe {nd::ArrayView::from_shape_ptr(shape, b.as_ptr())};
    let mut C: nd::ArrayBase<_, nd::Ix2>  = unsafe {nd::ArrayViewMut::from_shape_ptr(shape, c.as_mut_ptr())};

    
    bench.iter(|| {
        nd::linalg::general_mat_mul(1.0, &A, &B, 0.0, &mut C);
    })
}

unsafe fn malloc<T>(s: usize) -> *mut T {
    let l = alloc::Layout::array::<T>(s).unwrap();
    alloc::alloc(l) as *mut T
}

unsafe fn free<T>(s: usize, ptr: *mut T) {
    unsafe {
        alloc::dealloc(ptr as *mut u8, alloc::Layout::array::<T>(s).unwrap());
    }
}

fn bench_na(bench: &mut Bencher) {
    let shape_ = SHAPE;
    let elems = shape_.0 * shape_.1;
    let shape = (na::Dynamic::new(shape_.0), na::Dynamic::new(shape_.1));
    let stride = (na::Const::<1>, na::Dynamic::new(shape_.1));
    

    let a = unsafe {
        malloc::<f32>(elems)
    };
    let b = unsafe {
        malloc::<f32>(elems)
    };
    let c = unsafe {
        malloc::<f32>(elems)
    };
    let Ap = unsafe{ 
        na::SliceStorageMut::from_raw_parts(a, shape, stride) };
    let mut A = na::Matrix::from_data(Ap);
    
    let Bp = unsafe{ 
        na::SliceStorageMut::from_raw_parts(b, shape, stride) };
    let B = na::Matrix::from_data(Bp);

    let Cp = unsafe{ 
        na::SliceStorageMut::from_raw_parts(c, shape, stride) };
    let C = na::Matrix::from_data(Cp);

    bench.iter(|| {
        C.mul_to(&B, &mut A);
    });

    unsafe {
        free(elems, a);
        free(elems, b);
        free(elems, c);
    }
}



benchmark_group!(benches, bench_nd, bench_na);
benchmark_main!(benches);