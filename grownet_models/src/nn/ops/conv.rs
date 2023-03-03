use std::rc::Rc;

use af::{Dim4, Array};
use arrayfire::{self as af, dim4, HasAfEnum};
use super::{Param, Float, init, RcArray};

pub struct Conv2d<T: Float> {
    filter: Param<T>,
    bias: Option<Param<T>>,
    stride: [u64; 2],
    pad: [u64; 2],
}

impl<T: Float> Conv2d<T> {
    pub fn new(
        in_chan: u64, 
        out_chan: u64, 
        kernel_size: [u64; 2], 
        stride: [u64; 2],
        padding: [u64; 2],
        bias: bool,
    ) -> Self {
        let receptive_field = kernel_size[0] * kernel_size[1];
        let fan_in = receptive_field * in_chan;
        let fan_out = receptive_field * out_chan;
        let bias = if bias {
            Some(Param::new(init::Initializer::Zeros.init(dim4!(1, 1, out_chan, 1), fan_in, fan_out)))
        } else {
            None
        };
        Conv2d { 
            filter: Param::new(init::Initializer::HeNormal.init(dim4!(kernel_size[1], kernel_size[0], in_chan, out_chan), fan_in, fan_out)), 
            bias, 
            stride: stride, 
            pad: padding }
    }

    pub fn forward(&self, x: &Rc<Array<T>>) -> (Rc<Array<T>>, impl FnMut(&mut Self, &Array<T>) -> Array<T>) {
        let y = af::convolve2_nn(&x, &self.filter.w, 
            dim4!(self.stride[1], self.stride[0]), dim4!(self.pad[1], self.pad[0]), dim4!(1));
        
        let y = if let Some(x) = &self.bias {
            y + &x.w  
        } else {
            y
        };
        let y = Rc::new(y);
        let y1 = y.clone();
        let x1 = x.clone();
        let back_fn = move |s: &mut Conv2d<T>, grad: &Array<T>| {
            let dx = af::convolve2_gradient_nn(
                grad, &x1, &s.filter.w, &y1, 
                dim4!(s.stride[1], s.stride[0]), dim4!(s.pad[1], s.pad[0]), dim4!(1), af::ConvGradientType::DATA);
            let dw = af::convolve2_gradient_nn(
                grad, &x1, &s.filter.w, &y1, 
                dim4!(s.stride[1], s.stride[0]), dim4!(s.pad[1], s.pad[0]), dim4!(1), af::ConvGradientType::FILTER);
            s.filter.g += dw;
            if let Some(b) = &mut s.bias {
                let reordered = af::reorder_v2(&grad, 0, 1, Some(vec![3, 2]));
                let dim = reordered.dims();
                let flattened = af::moddims(&reordered, dim4!(dim[0] * dim[1] * dim[2], 1, dim[3]));
                let db = af::sum(&flattened, 0);
                b.g += db;
            }
            dx
        };
        (y, back_fn)
    }
}

#[test]
fn gradcheck_conv2d() {
    use super::utils::grad_check;
    use af::*;
    set_backend(Backend::CPU);
    let x = randn::<f64>(dim4!(24, 24, 3, 1));
    let w = randn::<f64>(dim4!(3, 3, 3, 8));
    let b = randn::<f64>(dim4!(1, 1, 8, 1));

    let stride = [2, 2];
    let pad = [1, 1];
    // let dilation = [1, 1];
    let fn_dx = |x: &Array<f64>| {
        let y = af::convolve2_nn(&x, &w, 
            dim4!(stride[1], stride[0]), dim4!(pad[1], pad[0]), dim4!(1));
        
        let y = add(&y, &b, true);
        let y1 = y.clone();
        let w1 = w.clone();
        let x1 = x.clone();
        let back = move |grad: &Array<f64>| {
            let dx = af::convolve2_gradient_nn(
                grad, &x1, &w1, &y1, 
                dim4!(stride[1], stride[0]), dim4!(pad[1], pad[0]), dim4!(1), af::ConvGradientType::DATA);
            dx
        };
        (y, back)
    };

    let fn_dw = |w: &Array<f64>| {
        let y = af::convolve2_nn(&x, &w, 
            dim4!(stride[1], stride[0]), dim4!(pad[1], pad[0]), dim4!(1));
        
        let y = add(&y, &b, true);
        let y1 = y.clone();
        let w1 = w.clone();
        let x1 = x.clone();
        let back = move |grad: &Array<f64>| {
            let dx = af::convolve2_gradient_nn(
                grad, &x1, &w1, &y1, 
                dim4!(stride[1], stride[0]), dim4!(pad[1], pad[0]), dim4!(1), af::ConvGradientType::FILTER);
            dx
        };
        (y, back)
    };

    let fn_db = |b: &Array<f64>| {
        let y = af::convolve2_nn(&x, &w, 
            dim4!(stride[1], stride[0]), dim4!(pad[1], pad[0]), dim4!(1));
        
        let y = add(&y, b, true);
        let back = move |grad: &Array<f64>| {
            let reordered = af::reorder_v2(&grad, 0, 1, Some(vec![3, 2]));
            let dim = reordered.dims();
            let flattened = af::moddims(&reordered, dim4!(dim[0] * dim[1] * dim[2], 1, dim[3]));
            let db = af::sum(&flattened, 0);
            //println!("{}", db.dims());
            db
        };
        (y, back)
    };


    grad_check(x.clone(), None, None, None, fn_dx);
    grad_check(w.clone(), None, None, None, fn_dw);
    grad_check(b.clone(), None, None, None, fn_db);
}