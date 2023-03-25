pub mod af_ops;
pub mod nd_ops;
pub mod parts;

pub use af_ops::Param;

use std::any::{Any, TypeId};

struct Test {
    a: i32,
    b: u32,
}

trait Trace: Sized + 'static {
    fn trace<T: 'static, F: FnMut(&T)>(&self, mut f: F) {
        let s: &dyn Any = &*self;
        if let Some(v) = s.downcast_ref::<T>() {
            f(v);
        }
    }
    fn trace_mut<T: 'static, F: FnMut(&mut T)>(&mut self, mut f: F) {
        let s: &mut dyn Any = &mut *self;
        if let Some(v) = s.downcast_mut::<T>() {
            f(v);
        }
    }
}

trait TracePath: 'static + Sized {
    fn trace_path<T: 'static, F: FnMut(&str, &T)>(&self, path: &str, mut f: F) {
        let s: &dyn Any = &*self;
        if let Some(v) = s.downcast_ref::<T>() {
            f(path, v);
        }
    }
    fn trace_path_mut<T: 'static, F: FnMut(&str, &mut T)>(&mut self, path: &str, mut f: F) {
        let s: &mut dyn Any = &mut *self;
        if let Some(v) = s.downcast_mut::<T>() {
            f(path, v);
        }
    }
}

impl Trace for Test {}

trait TraceV2<T, F: FnMut(&mut T)> {
    fn trace(&mut self, f: F);
}

impl<F: FnMut(&mut Test)> TraceV2<Test, F> for Test {
    fn trace(&mut self, _f: F) {

    }
}


#[test]
fn test() {
    let a = Test { a: 1, b: 2 };
    a.trace(|_x: &Test| {

    });
    
}

