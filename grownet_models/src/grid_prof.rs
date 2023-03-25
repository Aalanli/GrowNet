pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}

fn main() {
    use ndarray::prelude::*;
    use model_lib::models::grid_like::{m0, m0ctx};
    use model_lib::nn::nd_ops::context::{FlatCtx, NaiveCtx, BlockCtx};

    let xyz = black_box([32, 32, 32]);
    let batch_size = black_box(4);
    let dim = black_box(8);
    let iters = black_box(100);
    
    let xs = Array4::<f32>::zeros((xyz[1], xyz[0], batch_size, dim));
    let mut grid = m0::SimpleGrid::new(xyz, batch_size, dim);

    let mut y = 0.0;
    for i in 0..iters {
        let out = grid.forward(&xs.view());
        y += out[(0, 0, 0, 0)];
    }

    println!("{}", y);
}