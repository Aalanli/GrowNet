
/// Simplest computation block, a differentiable function
pub trait Node {
    type M;
    fn forward(&mut self, msg: Self::M) -> Self::M;
    fn update(&mut self, msg: Self::M) -> Self::M;
}

