pub mod m0;
pub mod m0ctx;

/// Simplest computation block, a differentiable function, whose update method must strictly follow the forward method
pub trait Node {
    type M;
    fn forward(&mut self, msg: Self::M) -> Self::M;
    fn update(&mut self, msg: Self::M) -> Self::M;
}


/// A more general node, where the forward takes in multiple inputs, outputting one result, and the backward does the
/// opposite
pub trait NodeV2 {
    type M;
    type R;
    fn forward(&mut self, inputs: Self::M) -> Self::R;
    fn backward(&mut self, inputs: Self::M, grad: Self::R, dinputs: Self::M);
}

/// An even more general node, where at each step, each node takes in some inputs and computes some outputs
/// updating ys, and takes in some derivates and updates d_xs
pub trait NodeV3 {
    type F; // input buffer
    type B; // output buffer
    fn step(&mut self, xs: Self::F, ys: Self::B, d_ys: Self::F, d_xs: Self::B);
}

/// An even more general node than the previous, at each step, each node takes in some inputs and computes some
/// outputs, arbitrarily, does not necessarily have to differentiate between forward and backwards passes, gradients
/// vs forward propagation. 
pub trait NodeV4 {
    type F;
    type B;
    fn step(&mut self, input: Self::F, output: Self::B);
}

/// It is the node's responsibility to adjust itself, hence the only functions it should support
/// are forward, of which a node receives a message, and decides if it should send one out
/// and backward, of which a node receives a push, and decides if it should send a push
/// in the future it may be advantageous to add a grow function, or just use higher abstractions such as NodeV4
pub trait NodeV5<Ctx> {
    type F;
    type B;
    fn forward(&mut self, ctx: Ctx, forward: Self::F) -> Option<Self::F>; // inputs and outputs should be homogeneous
    fn backward(&mut self, ctx: Ctx, backward: Self::B) -> Option<Self::B>; // backward messages should also be homogeneous
}

