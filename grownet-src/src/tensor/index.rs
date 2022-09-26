use std::ops::Deref;

pub trait TIndex {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize>;
}
struct UnsafeIndex<T>(T);

impl<T> Deref for UnsafeIndex<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TIndex for usize {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        if *self >= dims[0] * strides[0] {
            return None;
        }
        Some(*self)
    }
}

impl TIndex for UnsafeIndex<usize> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        Some(self.0)
    }
}

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

impl TIndex for Vec<usize> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        tile_cartesian(self, dims, strides)
    }
}

impl TIndex for &Vec<usize> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        tile_cartesian(self, dims, strides)
    }
}

impl<const N: usize> TIndex for [usize; N] {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        static_tile_cartesian::<N>(self, dims, strides)
    }
}

impl<const N: usize> TIndex for &[usize; N] {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        static_tile_cartesian::<N>(self, dims, strides)
    }
}

impl TIndex for UnsafeIndex<Vec<usize>> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        in_bounds_tile_cartesian(self, dims, strides)
    }
}

impl TIndex for UnsafeIndex<&Vec<usize>> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        in_bounds_tile_cartesian(self, dims, strides)
    }
}

impl<const N: usize> TIndex for UnsafeIndex<[usize; N]> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        in_bounds_static_tile_cartesian::<N>(self, dims, strides)
    }
}

impl<const N: usize> TIndex for UnsafeIndex<&[usize; N]> {
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize]) -> Option<usize> {
        in_bounds_static_tile_cartesian::<N>(self, dims, strides)
    }
}
