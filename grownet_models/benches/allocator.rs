use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn allocation_test(c: &mut Criterion) {
    use model_lib::ctx::allocator::{Alloc, FlatAllocator};

    let iteration = black_box(1000);
    let size = black_box(64);
    c.bench_function("naive allocation", |b| {
        b.iter(|| { 
            for _ in 0..iteration {
                let mut v = Vec::<f32>::new();
                v.reserve_exact(size);
            }
         })
    });

    let mut falloc = FlatAllocator::new(iteration * size * 12);
    c.bench_function("flat alloc", |b| {
        b.iter(|| {
            for _ in 0..iteration {
                let mut _slice = falloc.request_slice::<f32>(size).unwrap();
            }
            falloc.clear();
        });
    });

    let mut alloc = Alloc::new();
    c.bench_function("alloc", |b| {
        b.iter(|| {
            for _ in 0..iteration {
                let mut _slice = alloc.request_slice::<f32>(size);
            }
            alloc.clear();
        });
    });
}

pub fn elementwise_math(c: &mut Criterion) {
    use ndarray::prelude::*;
    use model_lib::ctx::*;
    let iteration = black_box(1000);
    let shape1 = black_box((64, 32));
    let shape2 = black_box((64, 32));

    c.bench_function("naive elemwise", |b| {
        b.iter(|| {
            let a = Array2::<f32>::zeros(shape1);
            let b = Array2::<f32>::ones(shape2);
            let mut out = Array2::<f32>::zeros(shape2);
            for _ in 0..iteration {
                out += &((&a + &b) * &a);
            }
            out
         })
    });

    let mut alloc = ArrayAlloc::new();
    c.bench_function("fused elemwise", |b| {
        b.iter(|| {
            let a = alloc.request::<f32, _, _>(shape1).zeros().id();
            let b = alloc.request::<f32, _, _>(shape2).ones().id();
            let mut out = Array2::<f32>::zeros(shape2);
            for _ in 0..iteration {
                let exec = (a + b) * a;
                out += &*exec.exec(&alloc).unwrap().view();
            }
            alloc.clear();
            out
        });
    });
}

criterion_group!(allocation, allocation_test);
criterion_group!(elementwise, elementwise_math);

criterion_main!(elementwise, allocation);