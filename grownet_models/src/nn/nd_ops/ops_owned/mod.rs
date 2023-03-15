use super::*;
pub mod norm;
pub use norm::*;

pub fn dmatmul<T: Float + 'static>(grad: &Array2<T>, a: &Array2<T>, b: &Array2<T>) -> (Array2<T>, Array2<T>) {
    let db = a.t().dot(grad);
    let da = grad.dot(&b.t());
    (da, db)
}

pub fn dot_axis<A: Float, D: Dimension + RemoveAxis>(x: &ArrayView<A, D>, y: &ArrayView<A, D>, axis: usize) -> Array<A, D> {
    let mut buf = unit_axis(x.raw_dim(), axis);
    for (view_a, view_b) in x.axis_iter(Axis(axis)).zip(y.axis_iter(Axis(axis))) {
        for ((z, a), b) in buf.iter_mut().zip(view_a.iter()).zip(view_b.iter()) {
            *z = a.mul_add(*b, *z);
        }
    }
    buf
}

pub fn mean_axis<A: Float, D: Dimension + RemoveAxis>(x: &ArrayView<A, D>, axis: usize) -> Array<A, D> {
    let mut buf = unit_axis(x.raw_dim(), axis);
    let n = A::from(x.len_of(Axis(axis))).unwrap();
    for view_a in x.axis_iter(Axis(axis)) {
        for (z, a) in buf.iter_mut().zip(view_a.iter()) {
            *z = *z + *a;
        }
    }
    buf.mapv_into(|x| x / n)
}


pub fn randn<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Array<f32, D> {
    Array::random(shape, Normal::new(0.0, 1.0).unwrap())
}

pub fn randn64<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Array<f64, D> {
    Array::random(shape, Normal::new(0.0, 1.0).unwrap())
}


pub fn unit_axis<D: Dimension, F: Float>(mut dim: D, i: usize) -> Array<F, D> {
    dim.slice_mut()[i] = 1;
    Array::zeros(dim)
}
