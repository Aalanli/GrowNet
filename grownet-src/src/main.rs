#![allow(dead_code)]

mod m2;
mod tensor;

use tensor as ts;

trait Test {
    fn test<T>(a: T) -> T;
}


fn main() {
    let slice = ts::tslice![1..2];
    let mut tensor = ts::WorldTensor::<i32>::new(vec![3, 3, 3]);
    let c = 1;
    tensor[&c] += 2;
    let mut sts: ts::MutWorldSlice<'_, i32> = ts::MutWorldSlice::<i32>::new(&mut tensor, slice);
    sts[&c] += 1;
    let h: i32 = sts[&c];
    println!("{}", h);
}
