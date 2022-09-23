fn test<'a, T>(x: &'a [T]) -> usize {
    x.len()
}

struct Test {
    a: i32,
    b: i64
}

fn main() {
    let a = Test{a: 1, b:2};
    let ptr: *const Test;
    ptr = &a as *const Test;
    let c = ptr as *mut Test;

    unsafe {
        (*c).a = 12;
    }

    print!("{}", a.a);

    let k = 1;
}
