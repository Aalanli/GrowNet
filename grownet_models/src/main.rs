use arrayfire::*;
use model_lib::models::baselinev2::FastResnet;

fn main() {
    let dim = 28;
    let n = 5;
    set_backend(Backend::CPU);
    let x = randn!(dim, dim, 3, 2);
    let mut resnet = FastResnet::new(10);

    let (y, df) = resnet.forward(&x);
    let _grad = df(&mut resnet, &y);

    use std::time::Instant;
    let inst = Instant::now();
    for _ in 0..n {
        let (y, df) = resnet.forward(&x);
        let _grad = df(&mut resnet, &y);
        _grad.eval();
    }

    println!("avg {} sec/step", inst.elapsed().as_secs_f32() / n as f32);
}