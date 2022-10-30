include("compute.jl")


mutable struct BaselineGrid2D{
    N<:AbstractNode, Msg<:AbstractMsg, GradMsg<:AbstractMsg, Ctx<:AbstractCtx} <: AbstractGrid

    nodes::Array{N, 2}
    node_ctx::Array{Ctx, 2}
    sum_buf1::Array{Msg, 1}
    sum_buf2::Array{Msg, 1}
    gsum_buf1::Array{GradMsg, 1}
    gsum_buf2::Array{GradMsg, 1}
    neighbour_offsets::NTuple{3, Int64}
end

function BaselineGrid2D(dims::NTuple{2, Int}, ::Type{Node}) where {Node <: AbstractNode}
    d = dims[1]
    offsets = (d - 1, d, d + 1)
    nodes = Array{Node, 2}(undef, dims)
    node_ctx = Array{ctx_type(Node), 2}(undef, dims)
    sum_buf1 = Array{msg_type(Node), 1}(undef, d + 2)
    sum_buf2 = Array{msg_type(Node), 1}(undef, d + 2)
    gsum_buf1 = Array{grad_type(Node), 1}(undef, d + 2)
    gsum_buf2 = Array{grad_type(Node), 1}(undef, d + 2)
    BaselineGrid2D{Node, msg_type(Node), grad_type(Node), ctx_type(Node)}(
        nodes, 
        node_ctx,
        sum_buf1,
        sum_buf2,
        gsum_buf1,
        gsum_buf2,
        offsets
    )
end