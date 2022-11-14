struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    #import bevy_pbr::mesh_vertex_output
};

struct VecMaterial {
    color: vec4<f32>
};

@group(1) @binding(0)
var<uniform> uniform_data: VecMaterial;


@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    return uniform_data.color;
}