use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::ops::Mul;
use num::Float;

pub fn fn_impl<T: Float>(f: impl Fn(T, T) -> T, a: T, b: T) -> T {
    f(a, b)
}

pub fn fn_ptr_impl<T: Float>(f: fn(T, T) -> T, a: T, b: T) -> T {
    f(a, b)
}


pub fn f00(a: i32) -> f32 {
    a as f32
}

pub fn f01(a: f32) -> f32 {
    a.mul(2.0).sin().cos()
}

#[inline(always)]
pub fn f02(a: f32, b: f32) -> f32 {
    ((a + b).sin().abs() + 1e-5).ln()
}


pub fn fn_ptr_test(c: &mut Criterion) {
    let it = black_box(1_000_000);
    let f0: fn(i32) -> f32 = |a| a as f32;
    let f1: fn(f32) -> f32 = |a| a.mul(2.0).sin().cos();
    let f2: fn(f32, f32) -> f32 = |a, b| ((a + b).sin().abs() + 1e-5).ln();
    c.bench_function("fn ptr impl", |b| {
        b.iter(|| {
            (0..it).map(f0).map(f1).fold(1.0, f2)
        });
    });

    c.bench_function("fn ptr inline", |b| {
        b.iter(|| {
            (0..it).map(f00).map(f01).fold(1.0, f02)
        });
    });

    c.bench_function("fn impl", |b| {
        b.iter(|| {
            let f0 = |a| a as f32;
            let f1 = |a: f32| -> f32 { a.mul(2.0).sin().cos() };
            let f2 = |a: f32, b: f32| ((a + b).sin().abs() + 1e-5).ln();
            (0..it).map(f0).map(f1).fold(1.0, f2)
        });
    });
}

pub struct ApplierPtr {
    f: fn(f32, f32) -> f32
}

impl ApplierPtr {
    fn apply(&self, it: impl Iterator<Item = f32>) -> f32 {
        it.fold(0.1, self.f)
    }
}

pub struct Applier<F> {
    f: F
}

impl<F: Fn(f32, f32) -> f32> Applier<F> {
    fn apply(&self, it: impl Iterator<Item = f32>) -> f32 {
        it.fold(0.1, &self.f)
    }
}

pub fn fn_applier_test(c: &mut Criterion) {
    let it = black_box(1_000_000);
    let f2: fn(f32, f32) -> f32 = |a, b| ((a + b).sin().abs() + 1e-5).ln();
    c.bench_function("applier fn ptr impl", |b| {
        let applier = ApplierPtr { f: f2 };
        b.iter(|| {
            applier.apply((0..it).map(|x| x as f32))
        });
    });

    c.bench_function("applier fn ptr inline", |b| {
        let applier = ApplierPtr { f: f02 };
        b.iter(|| {
            applier.apply((0..it).map(|x| x as f32))
        });
    });

    c.bench_function("applier fn impl", |b| {
        let applier = Applier { f: |a, b| a + b };
        b.iter(|| {
            applier.apply((0..it).map(|x| x as f32))
        });
    });
}


criterion_group!(fn_ptr, fn_ptr_test);
criterion_group!(fn_applier, fn_applier_test);
criterion_main!(fn_ptr, fn_applier);