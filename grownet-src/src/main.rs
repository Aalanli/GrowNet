#![allow(dead_code)]

//mod m2;
mod tensor;

use tensor as ts;

trait Test {
    fn test<T>(a: T) -> T;
}


fn main() {
    let slice = ts::tslice![.., 1..2];
    println!("{}", slice);
    let mut tensor = ts::WorldTensor::<i32>::new(vec![3, 3, 3]);
    let c = 1;
    let mut sts: ts::MutWorldSlice<'_, i32> = ts::MutWorldSlice::<i32>::new(&mut tensor, slice);
    println!("{:?}", sts.slice);
    sts[&[1,2]] += 9;
    //sts[&8] = 3;
    //tensor[&23] = 2;
    println!("{}", tensor);
}
