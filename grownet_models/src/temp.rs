#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = std::ptr::read_volatile(&dummy);
        std::mem::forget(dummy);
        ret
    }
}
fn main() {
    use ndarray::prelude::*;
    use model_lib::ctx::*;
    let outer = black_box(100);
    let iteration = black_box(1000);
    let shape1 = black_box((64, 32));
    let shape2 = black_box((64, 32));

    // let mut alloc = ArrayAlloc::new();
    // let mut out = Array2::<f32>::zeros(shape2);
    // for _ in 0..outer {
    //     let a = alloc.request::<f32, _, _>(shape1).zeros().id();
    //     let b = alloc.request::<f32, _, _>(shape2).ones().id();
    //     for _ in 0..iteration {
    //         let exec = (a + b) / a * a;
    //         out += &*exec.exec(&alloc).view();
    //     }
    //     alloc.clear();
    // }
    // println!("{}", out[(0, 0)]);

    let mut out = Array2::<f32>::zeros(shape2);
    for _ in 0..outer {
        let a = Array2::<f32>::zeros(shape1);
        let b = Array2::<f32>::ones(shape2);
        for _ in 0..iteration {
            out += &((&a + &b) / &a * &a);
        }
    }
    println!("{}", out[(0, 0)]);

}