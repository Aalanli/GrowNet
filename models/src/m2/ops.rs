use ndarray::prelude::*;
use num::{self, Float};


pub fn l2_norm<T: Float>(a: &Array1<T>) -> T {
    a.fold(T::zero(), |acc, x| acc + *x * *x).sqrt()
}


pub struct RunningStats {
    window: Vec<f32>,
    moving_avg: f32,
    window_size: u32,
    idx: u32
}

impl RunningStats {
    pub fn new(window: u32) -> Self {
        let mut v = Vec::<f32>::new();
        v.reserve_exact(window as usize);
        RunningStats {
            window: v, moving_avg: 0.0, window_size: window, idx: 0
        }
    }
    pub fn new_stat(&mut self, mag: f32) -> f32 {
        if self.window.len() <= self.window_size as usize {
            self.window.push(mag);
            self.moving_avg += mag;
            return self.moving_avg / self.window.len() as f32;
        } else {
            self.idx %= self.window_size;
            let last_elem = self.window[self.idx as usize];
            self.moving_avg += mag - last_elem;
            self.window[self.idx as usize] = mag;
            self.idx += 1;
            return self.moving_avg / self.window_size as f32;
        }
    }
}


// TODO: this should be a macro
/*
const fn get_neighbours(n: usize, t: usize) -> usize {
    t.pow(n as u32)
}

const fn gen_neighbourhood<const T: usize, const D: usize, const N: usize>(toggle: &[i32; T]) -> [[i32; D]; N] {
    let h = [[0; D]; N];
    map_multiple(h, 1, 0, toggle)
}

const fn map_multiple<const T: usize, const D: usize, const N: usize>(
    h: [[i32; D]; N], stride: usize, j: usize, toggle: &[i32; T]) -> [[i32; D]; N]
{
    if j == D {
        return h;
    }
    let h = map_test::<T, D, N>(h, j, stride, 0, toggle);
    map_multiple(h, stride * T, j + 1, toggle)
}

const fn map_test<const T: usize, const D: usize, const N: usize>(
    mut h: [[i32; D]; N], j: usize, stride: usize, idx: usize, toggle: &[i32; T]) -> [[i32; D]; N] 
{
    if idx == N {
        return h;
    }
    let p: usize = idx / stride;
    let toggle_idx = p % T;
    h[idx][j] = toggle[toggle_idx];
    return map_test::<T, D, N>(h, j, stride, idx + 1, toggle);
}

const T: usize = 3;
const TOGGLE: [i32; T] = [0, -1, 1];
const D: usize = 2;
const N: usize = get_neighbours(D, T);
const H: [[i32; D]; N] = gen_neighbourhood::<T, D, N>(&TOGGLE);
*/


#[cfg(test)]
mod test {
    #[test]
    fn underflow() {
        for i in 0..88 {
            let x = i as f32;
            let y = 1.0 / (1.0 + x.exp());
            if !(y > 0.0) {
                panic!("overflow at {i}");
            }
        }
    }
}