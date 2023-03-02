// taken from https://github.com/srenevey/neuro with slight modifications
use std::rc::Rc;

use arrayfire::*;
use super::Float;

pub struct MaxPool2D {
    kernel_size: [u64; 2],
    stride: [u64; 2]
}

impl MaxPool2D {
    pub fn new(kernel_size: [u64; 2], stride: [u64; 2]) -> Self {
        MaxPool2D {
            kernel_size, stride
        }
    }
    
    fn compute_output_shape(&self, input_shape: Dim4) -> Dim4 {
        let output_height = ((input_shape.get()[0] - self.kernel_size[0]) as f64 / self.stride[0] as f64 + 1.).floor() as u64;
        let output_width = ((input_shape.get()[1] - self.kernel_size[0]) as f64 / self.stride[1] as f64 + 1.).floor() as u64;
        Dim4::new(&[output_height, output_width, input_shape.get()[2], input_shape.get()[3]])
    }

    fn max_pool<T: Float>(&self, input: &Array<T>) -> (Array<T>, Array<i32>, Array<i32>, Dim4) {
        let cols = unwrap(input, self.kernel_size[0] as i64, self.kernel_size[1] as i64, self.stride[0] as i64, self.stride[1] as i64, 0, 0, true);
        let cols_reshaped = moddims(&cols, Dim4::new(&[cols.dims().get()[0], cols.elements() as u64 / cols.dims().get()[0], 1, 1]));

        // Computes max values and indices
        let (mut max_values, row_indices_u32) = imax(&cols_reshaped, 0);

        let output_shape = self.compute_output_shape(input.dims());
        // Creates the output
        let output = moddims(&max_values, Dim4::new(&[output_shape.get()[0], output_shape.get()[1], input.dims().get()[2], input.dims().get()[3]]));

        // Creates rows and columns indices
        let mut row_indices: Array<i32> = row_indices_u32.cast();
        //row_indices = reorder(&row_indices, Dim4::new(&[1, 0, 2, 3]));
        row_indices = reorder_v2(&row_indices, 1, 0, Some(vec![2, 3]));

        //max_values = reorder(&max_values, Dim4::new(&[1, 0, 2, 3]));
        max_values = reorder_v2(&max_values, 1, 0, Some(vec![2, 3]));
        let num_cols = max_values.dims().get()[0];
        let col_indices_vec: Vec<i32> = (0..num_cols as i32).collect();
        let mut col_indices = Array::new(&col_indices_vec[..], Dim4::new(&[num_cols, 1, 1, 1]));
        col_indices = tile(&col_indices, Dim4::new(&[cols_reshaped.dims().get()[2] * cols_reshaped.dims().get()[3], 1, 1, 1]));

        (output, row_indices, col_indices, output_shape)
    }

    pub fn forward<T: Float>(&self, x: &Rc<Array<T>>) -> (Rc<Array<T>>, impl FnMut(&Self, &Array<T>) -> Array<T>) {
        let (output, row_ind, col_ind, _) = self.max_pool(x);

        let row_ind = Rc::new(row_ind);
        let col_ind = Rc::new(col_ind);
        let back_fn = move |s: &Self, grad: &Array<T>| {
            let batch_size = grad.dims().get()[3];
            let flat_input = flat(&*grad);
            let sparse = sparse(s.kernel_size[0] * s.kernel_size[1], grad.elements() as u64, &flat_input, &row_ind, &col_ind, SparseFormat::COO);
            let mut dense = sparse_to_dense(&sparse);
            let num_channels = grad.dims()[2];
            let num_cols = dense.dims().get()[1] / (num_channels * batch_size);
            dense = moddims(&dense, Dim4::new(&[dense.dims().get()[0], num_cols, num_channels, batch_size]));
            wrap(&dense, grad.dims().get()[0] as i64, grad.dims().get()[1] as i64, s.kernel_size[0] as i64, s.kernel_size[1] as i64, s.stride[0] as i64, s.stride[1] as i64, 0, 0, true)
        };

        (Rc::new(output), back_fn)
    }
}