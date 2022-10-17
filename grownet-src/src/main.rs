#![allow(dead_code)]

mod m2;
mod tensor;

const fn get_neighbours(n: usize, t: usize) -> usize {
    t.pow(n as u32)
}

const fn gen_neighbourhood<const T: usize, const D: usize, const N: usize>(toggle: &[i32; T]) -> [[i32; D]; N] {
    let h = [[0; D]; N];
    map_multiple(h, 1, 0, toggle)
}

const fn map_multiple<const T: usize, const D: usize, const N: usize>(
    h: [[i32; D]; N], stride: usize, j: usize, toggle: &[i32; T]) -> [[i32; D]; N]
{
    if j == D {
        return h;
    }
    let h = map_test::<T, D, N>(h, j, stride, 0, toggle);
    map_multiple(h, stride * T, j + 1, toggle)
}

const fn map_test<const T: usize, const D: usize, const N: usize>(
    mut h: [[i32; D]; N], j: usize, stride: usize, idx: usize, toggle: &[i32; T]) -> [[i32; D]; N] 
{
    if idx == N {
        return h;
    }
    let p: usize = idx / stride;
    let toggle_idx = p % T;
    h[idx][j] = toggle[toggle_idx];
    return map_test::<T, D, N>(h, j, stride, idx + 1, toggle);
}

const fn linear_dimensional_offset<const T: usize, const D: usize>(h: [[i32; D]; N], strides: [i32; D]) {

}


const T: usize = 3;
const TOGGLE: [i32; T] = [0, -1, 1];
const D: usize = 2;
const N: usize = get_neighbours(D, T);
const H: [[i32; D]; N] = gen_neighbourhood::<T, D, N>(&TOGGLE);


fn main() {
    println!("{}", N);
    for i in 0..N {
        for j in 0..D {
            print!("{} ", H[i][j]);
        }
        println!("");
    }
}
