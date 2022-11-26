pub mod compute_matrix;
pub mod compute_node;
pub mod activations;
pub mod ops;

use ndarray::Array1;
use std::ops::Deref;
use ops::l2_norm;

pub use compute_node::ComputeNode;

/// The global setup parameters for the entire network,
/// is passed by reference into each forward and backward call
#[derive(Clone)]
pub struct GlobalParams {
    sim_strictness: f32,
    underflow_epsilon: f32,
    sim_epsilon: f32,
    lr: f32,
    compute_dim: usize,

    s_mean: f32,
    s_var: f32,
    b_mean: f32,
    b_var: f32,

    ro_var: f32,
}

/// The main enum signifying the output state of a node, as nodes may
/// not output anything if internal requirements are not met
#[derive(Clone)]
pub enum NodeResult<T: Clone> {
    NoResult,
    Msg(T),
}

/// Primary message format in the network, probably should make this
/// polymorphic with respect number of dimensions
#[derive(Clone)]
pub struct Message(Array1<f32>);

/// Primary messages passed between different nodes, should make this a polymorphic
/// trait
#[derive(Clone)]
pub struct NodeMessage {
    msg: Message,
    mag: f32
}



impl Deref for Message {
    type Target = Array1<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0        
    }
}

impl From<Array1<f32>> for Message {
    fn from(a: Array1<f32>) -> Self {
        Message(a)
    }
}

impl NodeMessage {
    pub fn new(msg: Message) -> NodeMessage {
        let norm = l2_norm(&msg);
        NodeMessage { msg, mag: norm }
    }
}

