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
    let mut sts: ts::MutWorldSlice<'_, i32> = ts::MutWorldSlice::<i32>::new(&mut tensor, &slice);
    println!("{:?}", sts.slice);
    
    sts.iter_mut().for_each(|x| {*x = -1;});
    println!("hello {}", tensor);
}
