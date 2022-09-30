#![allow(dead_code)]


mod m2;
mod tensor;

use tensor as ts;


fn main() {
    let slice = ts::tslice![1..2];
    let ts = ts::WorldTensor::<i32>::new(vec![3, 3, 3]);
    let h = ts[[1, 2, 1]];
    println!("{}", h);
}
