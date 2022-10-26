mod compute;
mod grid;
mod indexing;

pub struct GlobalParams {
    lr: f32,
    compute_dim: usize,
    grid_dim: Vec<usize>
}