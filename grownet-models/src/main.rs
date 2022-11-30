use ndarray_rand::RandomExt;
use rand_distr::Normal;
use tch::Tensor;
use ndarray::prelude::*;

fn main() {
    let a = Tensor::rand(&[4, 3, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::rand(&[3, 3, 3, 3], (tch::Kind::Float, tch::Device::Cpu));
    let b = a.conv2d::<Tensor>(&k, None, &[2, 2], &[0, 0], &[1, 1], 1);
    let h = a.conv2d::<Tensor>(&k, None, &[2, 2], &[0, 0], &[1, 1], 1);
    let d = a.conv2d::<Tensor>(&k, None, &[2, 2], &[0, 0], &[1, 1], 1);
    let afg = (&b) * (&h);
    println!("{}", b);

    let mut a = Array2::<f32>::random((128, 128), Normal::new(0.0, 1.0).unwrap());
    let mut b = Array2::<f32>::ones((128, 128));
    let mut c = Array2::<f32>::zeros((128, 128));

    c += &b;
    let e = a.view_mut();
    let f = b.view_mut();
    let mut g = c.view_mut();
    g += &f;
}
