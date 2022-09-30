use std::ops::Deref;

pub trait TIndex {
    type MP: TIndex;
    fn tile_cartesian(&self, dims: &[usize], strides: &[usize], len: usize) -> Option<usize>;
    fn offset(&self, offsets: &[usize], strides: &[usize]) -> Self::MP;
}

pub struct UnsafeIndex<T>(T);

impl<T> Deref for UnsafeIndex<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn check_index(index: usize, len: usize) -> Option<usize> {
    if index > len {
        return None;
    }
    Some(index)
}

fn tile_cartesian(indices: &[usize], dims: &[usize], strides: &[usize], _len: usize) -> Option<usize> {
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

fn in_bounds_tile_cartesian(indices: &[usize], dims: &[usize], strides: &[usize], _len: usize) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - indices.len();
    for i in 0..indices.len() {
        let i = i + offset;
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn static_tile_cartesian<const N: usize>(indices: &[usize; N], dims: &[usize], strides: &[usize], _len: usize) -> Option<usize> {
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

fn in_bounds_static_tile_cartesian<const N: usize>(indices: &[usize; N], dims: &[usize], strides: &[usize], _len: usize) -> Option<usize> {
    let mut idx: usize = 0;
    let offset = dims.len() - N;
    for i in 0..N {
        let i = i + offset;
        idx += strides[i] * indices[i];
    }
    Some(idx)
}

fn offset_ind(indices: &[usize], offsets: &[usize], _strides: &[usize]) -> Vec<usize> {
    let mut new_indices: Vec<usize> = indices.to_vec();
    for i in 0..offsets.len() {
        new_indices[i] += offsets[i];
    }
    new_indices
}

fn static_offset_ind<const N: usize>(indices: &[usize; N], offsets: &[usize], _strides: &[usize]) -> [usize; N] {
    let mut new_indices = indices.clone();
    for i in 0..offsets.len() {
        new_indices[i] += offsets[i];
    }
    new_indices
}

fn inbounds_retile_cart(index: usize, s_strides: &[usize], g_strides: &[usize], _len: usize) -> Option<usize> {
    let mut idx = 0;
    let mut ind = index;
    for i in 0..s_strides.len() {
        let c = ind / s_strides[i] as usize;
        idx += c * g_strides[i];
        ind -= c * s_strides[i];
        if ind <= 0 {
            break;
        }
    }
    Some(idx)
}

fn retile_cart(index: usize, s_strides: &[usize], g_strides: &[usize], len: usize) -> Option<usize> {
    if let Some(idx) = inbounds_retile_cart(index, s_strides, g_strides, len) {
        if idx > len {
            return None;
        }
        Some(idx);
    }
    None
}

macro_rules! RepeatFn {
    ($op_fn:ident, $fn_name:ident($($arg_nm:ident : $tp:ty),*) -> $rt:ty) => {
        fn $fn_name(&self, $($arg_nm : $tp),*) -> $rt {
            ($op_fn)(self, $($arg_nm),*)
        }
    }
}

macro_rules! deriveTIndex {
    ($tp:ty, [$([$op_fn:ident, $fn_name:ident($($arg_nm:ident : $atp:ty),*) -> $rt:ty]),*]) => {
        impl TIndex for $tp {
            type MP = Vec<usize>;
            $(RepeatFn!($op_fn, $fn_name($($arg_nm : $atp),*) -> $rt);)*
        }
    };

    ($N:ident, $tp:ty, [$([$op_fn:ident, $fn_name:ident($($arg_nm:ident : $atp:ty),*) -> $rt:ty]),*]) => {
        impl<const $N:usize> TIndex for $tp {
            type MP = [usize; N];
            $(RepeatFn!($op_fn, $fn_name($($arg_nm : $atp),*) -> $rt);)*
        }
    };    
}


deriveTIndex!(Vec<usize>, [
    [tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(&Vec<usize>, [
    [tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(&[usize], [
    [tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(UnsafeIndex<Vec<usize>>, [
    [in_bounds_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(UnsafeIndex<&Vec<usize>>, [
    [in_bounds_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(UnsafeIndex<&[usize]>, [
    [in_bounds_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(N, [usize; N], [
    [static_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [static_offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(N, &[usize; N], [
    [static_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [static_offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(N, UnsafeIndex<[usize; N]>, [
    [in_bounds_static_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [static_offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);
deriveTIndex!(N, UnsafeIndex<&[usize; N]>, [
    [in_bounds_static_tile_cartesian, tile_cartesian(dims: &[usize], strides: &[usize], len: usize) -> Option<usize>],
    [static_offset_ind, offset(offsets: &[usize], strides: &[usize]) -> Self::MP]]);


// This code is an abomination
// I want to only use one trait to define two behaviors for usize
// If WorldTensor is indexed with usize, regular pointer arithmetic applies
// but if WorldSlice is indexed with usize, some tiling operation should
// occur. Since only WorldSlice calls offset, I implement an inner behavior

#[derive(Clone, Copy)]
pub struct RequireRetileWrapper<T>(T, *const usize, usize);

impl TIndex for RequireRetileWrapper<usize> {
    type MP = RequireRetileWrapper<usize>;
    fn offset(&self, _offsets: &[usize], _strides: &[usize]) -> Self::MP {
        *self
    }
    fn tile_cartesian(&self, _dims: &[usize], world_strides: &[usize], len: usize) -> Option<usize> {
        let slice_strides = unsafe {
            std::slice::from_raw_parts(self.1, self.2)
        };
        retile_cart(self.0, slice_strides, world_strides, len)
    }
}

impl TIndex for usize {
    type MP = RequireRetileWrapper<usize>;
    fn offset(&self, _offsets: &[usize], strides: &[usize]) -> Self::MP {
        RequireRetileWrapper(*self, strides.as_ptr(), strides.len())
    }
    fn tile_cartesian(&self, _dims: &[usize], _strides: &[usize], len: usize) -> Option<usize> {
        check_index(*self, len)
    }
}

impl TIndex for UnsafeIndex<usize> {
    type MP = RequireRetileWrapper<UnsafeIndex<usize>>;
    fn offset(&self, _offsets: &[usize], strides: &[usize]) -> Self::MP {
        RequireRetileWrapper(UnsafeIndex(self.0), strides.as_ptr(), strides.len())
    }
    fn tile_cartesian(&self, _dims: &[usize], _strides: &[usize], _len: usize) -> Option<usize> {
        Some(self.0)
    }
}

impl TIndex for RequireRetileWrapper<UnsafeIndex<usize>> {
    type MP = RequireRetileWrapper<UnsafeIndex<usize>>;
    fn offset(&self, _offsets: &[usize], strides: &[usize]) -> Self::MP {
        RequireRetileWrapper(UnsafeIndex(*self.0), strides.as_ptr(), strides.len())
    }
    fn tile_cartesian(&self, _dims: &[usize], world_strides: &[usize], len: usize) -> Option<usize> {
        let slice_strides = unsafe {
            std::slice::from_raw_parts(self.1, self.2)
        };
        inbounds_retile_cart(*self.0, slice_strides, world_strides, len)
    }
}
    