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