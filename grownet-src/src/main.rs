#![allow(dead_code)]


mod m2;
mod tensor;

use tensor as ts;

fn test<const N: usize>() {
    println!("");
}

fn main() {
    let a = [1, 2, 3];
    let b = &a;
    let c = b.iter();
}
