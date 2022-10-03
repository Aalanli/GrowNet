#![allow(dead_code)]

//mod m2;
mod tensor;

use tensor as ts;

trait Test {
    fn test<T>(a: T) -> T;
}


fn main() {
    let slice = ts::tslice![.., 1, 1];
    println!("{}", slice);
    let mut tensor = ts::WorldTensor::<i32>::new(vec![3, 3, 3]);
    let mut sts: ts::MutWorldSlice<'_, i32> = tensor.slice_mut(&slice);
    let slice2 = ts::tslice![1];
    println!("{:?}", sts.slice);
    let mut sts2 = sts.slice_mut(&slice2);
    println!("{:?}", sts2.slice);
    sts2[[0]] = 1;
    println!("hello");
    sts2.iter_mut().for_each(|x| {*x = -1;});
    println!("{}", tensor);
}
