use std::f32::consts::PI;

use ndarray::prelude as np;
use rand_distr::Distribution;

use super::super::tensor as ts;
use super::{ComputeNode, GlobalParams, Message, NodeMessage, NodeResult};
use NodeResult::NoResult;
use NodeResult::Msg;

pub struct ComputeMatrix2D {
    nodes: ts::WorldTensor<ComputeNode>,
    edges: ts::WorldTensor<NodeParams>,
    global_params: GlobalParams,
    dims: [usize; 2],
    neighbour_pos: [isize; 3],
    neighbour_dot: [f32; 3],
}

impl ComputeMatrix2D {
    // initial offset for indexing into edges, there is some padding involved
    const PAD_OFFSET: usize = 1;
    pub fn new(global_params: &GlobalParams, dims: [usize; 2]) -> Self {
        let vdims = dims.to_vec();
        let nodes = ts::WorldTensor::<ComputeNode>::new(vdims.clone(), || ComputeNode::new(global_params));
        // we offset the size of the edges by 2, since neighbours border 1 around
        // doing this saves checking each cell individually if the neighbour indices are inbounds 
        let edges = ts::WorldTensor::<NodeParams>::new(vec![dims[0] + 1, dims[1] + 2], 
        || NodeParams::new(global_params));
        

        let neighbour_pos = [-(dims[1] as isize) - 1, -(dims[1] as isize), -(dims[1] as isize) + 1];
        let angles = [PI / 4.0, 0.0, -PI / 4.0];

        ComputeMatrix2D { nodes, edges, global_params: global_params.clone(), dims, neighbour_pos, neighbour_dot: angles }
    }

    /// each node looks back to three of its past neighbours, inspects their outputs stored in self.edges
    /// and computes their accumulation and weightings, where NoResult represents that the past neuron did not fire 
    fn accum_one_node(&self, idx: usize) -> NodeResult<NodeMessage> {
        let mut sum: NodeResult<NodeMessage> = NoResult;
        for i in 0..self.neighbour_pos.len() {
            let connected = self.neighbour_pos[i];
            let angle = self.neighbour_dot[i];
            let ro = self.edges[idx + Self::PAD_OFFSET].ro;
            if let Msg(a) = &self.edges[((idx + Self::PAD_OFFSET) as isize + connected) as usize + self.dims[1] + 2].forward_selfmsg {
                let sim = NodeParams::similarity(ro, angle, self.global_params.sim_strictness);
                if let Msg(accum) = &mut sum {
                    accum.msg.0 += &(&a.msg.0 * sim);
                    accum.mag += a.mag * sim;
                } else {
                    let mut n = (*a).clone();
                    n.msg.0 *= sim;
                    n.mag *= sim;
                    sum = Msg(n);
                }
            }
        }
        sum
    }

    /// It is unfortunate that it has come to this, I wanted a more functional
    /// approach, but the semantics of the model enforced this, due to performance reasons.
    pub fn set_inputs(&mut self, msgs: InputMessage) {
        for (idx, msg) in msgs.0 {
            let msg = NodeMessage::new(msg);
            self.edges[idx + Self::PAD_OFFSET].forward_selfmsg = Msg(msg);
        }
    }

    /// Update the internal representations in one forward pass, and stops early if 
    /// all nodes in a particular dimension does not give an output
    pub fn try_forward(&mut self) -> NodeResult<OutputMessage> {
        let stride = self.dims[1] + 2;
        for row_id in 0..self.dims[0] {
            let row_edge_offset = row_id * stride;
            let mut has_connection = false;
            for col_id in 0..self.dims[1] {
                if let Msg(sum) = self.accum_one_node(row_edge_offset + col_id) {
                    self.edges[row_edge_offset + col_id + Self::PAD_OFFSET + stride].forward_selfmsg = 
                        self.nodes[row_edge_offset + col_id].forward(&sum, &self.global_params);
                    has_connection = true;
                }
            }
            if !has_connection {
                return NoResult;
            }
        }
        let mut out_msg = vec![];
        out_msg.reserve_exact(self.dims[1]);
        let last_dim = (self.dims[0] - 1) * (self.dims[1] + 2);
        for i in 0..self.dims[1] {
            out_msg.push(self.edges[i + last_dim].forward_selfmsg.clone());
        }
        Msg(OutputMessage(out_msg))
    }

    /// Relax or restrict the inner node thresolds, similar to the principle of local excitation and global
    /// suppression. Unfortunately, updating the thresholds would make the previous computed passes invalid
    /// for back-prop
    pub fn update_thresholds(&mut self) {
        
    }


    pub fn cache(&self) {}
    pub fn extract_output(&self) {}
}

pub struct InputMessage(Vec<(usize, Message)>);
#[derive(Clone)]
pub struct OutputMessage(Vec<NodeResult<NodeMessage>>);

pub struct NodeParams {
    pub ro: f32,
    pub forward_selfmsg: NodeResult<NodeMessage>,
}

impl NodeParams {
    pub fn new(global_params: &GlobalParams) -> Self {
        use rand::{thread_rng};
        use rand_distr::Normal;

        let mut rng = thread_rng();
        NodeParams { ro: Normal::new(0.0, global_params.ro_var).unwrap().sample(&mut rng),
            forward_selfmsg: NoResult }
    }
}

impl NodeParams {
    fn similarity(a: f32, b: f32, strictness: f32) -> f32 {
        strictness * ((a - b).cos() - 1.0).exp()
    }
}