use std::ops::Deref;

pub trait TIndex {
    type MP: TIndex;
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize>;
    fn offset(&self, offsets: &[usize]) -> Self::MP;
}

pub struct UnsafeIndex<T>(T);

impl<T> Deref for UnsafeIndex<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

//impl TIndex for usize {
//    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
//        if *self >= dims[0] * strides[0] {
//            return None;
//        }
//        Some(*self)
//    }
//}
//
//impl TIndex for UnsafeIndex<usize> {
//    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
//        Some(self.0)
//    }
//}

fn tile_cartesian(indices: &[usize], dims: &[usize], strides: &[usize]) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - indices.len();
    for i in 0..indices.len() {
        let i = i + offset;
        if indices[i] >= dims[i] {
            return None;
        }
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn in_bounds_tile_cartesian(indices: &[usize], dims: &[usize], strides: &[usize]) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - indices.len();
    for i in 0..indices.len() {
        let i = i + offset;
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn static_tile_cartesian<const N: usize>(indices: &[usize; N], dims: &[usize], strides: &[usize]) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - N;
    for i in 0..N {
        let i = i + offset;
        if indices[i] >= dims[i] {
            return None;
        }
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn in_bounds_static_tile_cartesian<const N: usize>(indices: &[usize; N], dims: &[usize], strides: &[usize]) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - N;
    for i in 0..N {
        let i = i + offset;
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn offset_ind(indices: &[usize], offsets: &[usize]) -> Vec<usize> {
    let mut new_indices: Vec<usize> = indices.to_vec();
    for i in 0..offsets.len() {
        new_indices[i] += offsets[i];
    }
    new_indices
}

fn static_offset_ind<const N: usize>(indices: &[usize; N], offsets: &[usize]) -> [usize; N] {
    let mut new_indices = indices.clone();
    for i in 0..offsets.len() {
        new_indices[i] += offsets[i];
    }
    new_indices
}


macro_rules! deriveTIndex {
    ($N:ident, $op_func:ident, $dv_func:ident, $tp:ty, $($args:ident),*) => {
        impl<const $N:usize> TIndex for $tp {
            type MP = [usize; N];
            fn $dv_func(&self, $($args : &[usize]),*) -> Option<usize> {
                ($op_func)(&(*self), $($args),*)
            }
            fn offset(&self, offsets: &[usize]) -> [usize; N] {
                static_offset_ind(&self, offsets)
            }
        }
    };

    ($op_func:ident, $dv_func:ident, $tp:ty, $($args:ident),*) => {
        impl TIndex for $tp {
            type MP = Vec<usize>;
            fn $dv_func(&self, $($args : &[usize]),*) -> Option<usize> {
                ($op_func)(&(*self), $($args),*)
            }
            fn offset(&self, offsets: &[usize]) -> Vec<usize> {
                offset_ind(&self, offsets)
            }
        }
    };
}

deriveTIndex!(tile_cartesian, tile_cartesian, Vec<usize>, dims, strides);
deriveTIndex!(N, static_tile_cartesian, tile_cartesian, [usize; N], dims, strides);
deriveTIndex!(in_bounds_tile_cartesian, tile_cartesian, UnsafeIndex<Vec<usize>>, dims, strides);
deriveTIndex!(in_bounds_tile_cartesian, tile_cartesian, UnsafeIndex<&Vec<usize>>, dims, strides);
deriveTIndex!(N, in_bounds_static_tile_cartesian, tile_cartesian, UnsafeIndex<&[usize; N]>, dims, strides);
