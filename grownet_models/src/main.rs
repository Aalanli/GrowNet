use af::{dim4, Array, eval};
use arrayfire::{self as af, *};

fn main() {
    af::set_backend(af::Backend::CPU);
    let mut a = randn::<f64>(dim4!(3, 3));
    a.eval();
    assign(&mut a, 0, 1.0);
    let b = compute(&a);
    assign(&mut a, 0, 1.0001);
    let c = compute(&a);
    af::print(&(b - c));
}

fn assign(a: &mut Array<f64>, i: usize, val: f64) -> f64 {
    assert!(a.get_backend() == Backend::CPU);
    
    unsafe {
        let ptr = a.device_ptr() as *mut f64;
        let orig = *ptr.add(i);
        *ptr.add(i) = val;
        a.unlock();
        orig
    }
}

fn compute(x: &Array<f64>) -> Array<f64> {
    sin(x)
}