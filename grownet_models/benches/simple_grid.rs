use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn allocation_test(c: &mut Criterion) {
    use ndarray::prelude::*;
    use model_lib::nn::nd_ops::{context::FlatCtx, ops_ctx::*};

    let size = black_box([32, 32]);
    let trials = black_box(512);
    let mut store = Vec::new();
    store.reserve_exact(trials);
    c.bench_function("naive alloc mat", |b| {
        b.iter(|| {
            let input = Array2::<f32>::zeros(size);
            let weight = Array2::<f32>::zeros(size);
            for _ in 0..trials {
                let new = input.dot(&weight);
                store.push(new);
            }
            store.clear();
        });
    });

    let mut ctx = FlatCtx::<f32>::new(10usize.pow(8));
    let mut store = Vec::new();
    store.reserve_exact(trials);
    c.bench_function("ctx alloc mat", |b| {
        b.iter(|| {
            let input = ctx.zeros(size);
            let weight = ctx.zeros(size);
            for _ in 0..trials {
                let new = matmul(&ctx, &input.view(), &weight.view());
                let id = ctx.id(new);
                store.push(id);
            }
            ctx.clear();
            store.clear();
        });
    });

    let mut store = Vec::new();
    store.reserve_exact(trials);
    c.bench_function("naive alloc add", |b| {
        b.iter(|| {
            let input = Array2::<f32>::zeros(size);
            let weight = Array2::<f32>::zeros(size);
            for _ in 0..trials {
                let new = &input + &weight;
                store.push(new);
            }
            store.clear();
        });
    });

    let mut ctx = FlatCtx::<f32>::new(10usize.pow(8));
    let mut store = Vec::new();
    store.reserve_exact(trials);
    c.bench_function("ctx alloc add", |b| {
        b.iter(|| {
            let input = ctx.zeros(size);
            let weight = ctx.zeros(size);
            for _ in 0..trials {
                let mut new = ctx.zeros(size);
                ndarray::Zip::from(&mut new).and(&input).and(&weight)
                    .for_each(|y, a, b| {
                        *y = *a + *b;
                    });
                let id = ctx.id(new);
                store.push(id);
            }
            ctx.clear();
            store.clear();
        });
    });

}

pub fn simple_grid_benchmark(c: &mut Criterion) {
    use ndarray::prelude::*;
    use model_lib::models::grid_like::{m0, m0ctx};
    use model_lib::nn::nd_ops::context::{FlatCtx, NaiveCtx, BlockCtx};

    let xyz = black_box([32, 32, 32]);
    let batch_size = black_box(4);
    let dim = black_box(8);
    
    let xs = Array4::<f32>::zeros((xyz[1], xyz[0], batch_size, dim));
    let mut grid = m0::SimpleGrid::new(xyz, batch_size, dim);

    c.bench_function("grid naive", |b| 
        b.iter(|| { grid.forward(&xs.view()); }));
    
    let mut ctx_temp = FlatCtx::<f32>::new(10usize.pow(8u32));
    let mut grid = m0ctx::SimpleGrid::new(xyz, dim);

    c.bench_function("grid ctx flat", |b| 
        b.iter(|| { grid.forward(&ctx_temp,&xs.view()); ctx_temp.clear(); }));

    // let mut ctx_temp = NaiveCtx::<f32>::new();
    // let mut grid = m0ctx::SimpleGrid::new(xyz, dim);

    // let mut group = c.benchmark_group("grid ctx naive");    
    // group.sample_size(10);
    // group.bench_function("grid ctx naive", |b| {
    //     b.iter(|| { grid.forward(&ctx_temp,&xs.view()); ctx_temp.clear(); })
    // });
    // group.finish();

    // let mut ctx_temp = BlockCtx::<f32>::new(10usize.pow(5u32));
    // let mut grid = m0ctx::SimpleGrid::new(xyz, dim);

    // c.bench_function("grid ctx block 10e5", |b| 
    //     b.iter(|| { grid.forward(&ctx_temp,&xs.view()); ctx_temp.clear(); }));
    
}

criterion_group!(allocator, allocation_test);
criterion_group!(simple_grid, simple_grid_benchmark);
criterion_main!(simple_grid, allocator);