use std::rc::Rc;

use arrayfire::*;

use super::{Param, Float, init};

pub struct Linear<T: Float> {
    w: Param<T>,
    bias: Option<Param<T>>
}

impl<T: Float> Linear<T> {
    pub fn new(in_dim: u64, out_dim: u64, bias: bool) -> Self {
        Self { 
            w: Param::new(init::Initializer::HeNormal.init(dim4!(out_dim, in_dim), in_dim, out_dim)), 
            bias: if bias {
                Some(Param::new(init::Initializer::Zeros.init(dim4!(out_dim), in_dim, out_dim)))
            } else { None } 
        }
    }

    /// expect x to be [in_dim, H, ...], outputs [out_dim, H, ...]
    pub fn forward(&self, x: &Array<T>) -> (Array<T>, impl FnMut(&mut Self, &Array<T>) -> Array<T>) {
        let y = matmul(&self.w.w, &x, MatProp::NONE, MatProp::NONE);
        let y = if let Some(b) = &self.bias {
            y + &b.w
        } else {
            y
        };
        let x1 = x.clone();
        let back_fn = move |s: &mut Linear<T>, grad: &Array<T>| {
            let dx = matmul(&s.w.w, grad, MatProp::TRANS, MatProp::NONE);
            let dw = matmul(grad, &x1, MatProp::NONE, MatProp::TRANS);
            s.w.g += dw;
            if let Some(b) = &mut s.bias {
                let dims = grad.dims();
                let db = moddims(grad, dim4!(dims[0], dims[1] * dims[2] * dims[3]));
                let db = sum(&db, 1);
                b.g += db;
            }
            dx
        };
        (y, back_fn)
    }
}

