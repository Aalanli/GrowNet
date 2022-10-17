use ndarray::prelude as np;
use rand_distr::Distribution;

use crate::tensor as ts;
use super::{ComputeNode, GlobalParams, Message, NodeMessage};

pub struct ComputeMatrix2D {
    nodes: ts::WorldTensor<ComputeNode>,
    edges: ts::WorldTensor<NodeParams>,
    global_params: GlobalParams,
    dims: [usize; 2]
}

impl ComputeMatrix2D {
    pub fn new(global_params: &GlobalParams, dims: [usize; 2], listen_nodes: Vec<[usize; 2]>) -> Self {
        let vdims = dims.to_vec();
        let mut nodes = ts::WorldTensor::<ComputeNode>::new(vdims.clone()); 
        nodes.iter_mut().for_each(|x| {*x = ComputeNode::new(&global_params); });
        let mut edges = ts::WorldTensor::<NodeParams>::new(vdims);
        edges.iter_mut().for_each(|x| {*x = NodeParams::new(global_params); });

        ComputeMatrix2D { nodes, edges, global_params: global_params.clone(), dims: dims }
    }

    pub fn forward(&mut self, msgs: Vec<Message>) {}
    pub fn backward(&mut self, msgs: Vec<Message>) {}

    pub fn cache(&self) {}
    pub fn extract_output(&self) {}
}

pub struct NodeActor {
    
}

pub struct MultiDimensionalIndex<const D: usize>([usize; D]);
// acts like a multidimensional index
pub struct TilingIndex<'a, const D: usize> {
    index: usize,
    dimension: &'a GenericDimensionConverter<D>
}

pub struct GenericDimensionConverter<const D: usize> {
    dimension: [usize; D],
    strides: [usize; D],

}


pub struct NodeParams {
    ro: f32
}

impl NodeParams {
    pub fn new(global_params: &GlobalParams) -> Self {
        use rand::{thread_rng};
        use rand_distr::Normal;

        let mut rng = thread_rng();
        NodeParams { ro: Normal::new(0.0, global_params.ro_var).unwrap().sample(&mut rng) }
    }
}

impl NodeParams {
    fn similarity(a: f32, b: f32, strictness: f32) -> f32 {
        strictness * ((a - b).cos() - 1.0).exp()
    }
}