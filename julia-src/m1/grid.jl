include("compute.jl")
import Base.similar

function zero!(a::Array{<:AbstractMsg})
    for i in eachindex(a)
        @inbounds zero!(a[i])
    end
end


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
    offsets = (-1, 0, 1)
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

function populate_all!(a::Array{S}, init_fn) where {S}
    for i in CartesianIndices(a)
        a[i] = init_fn(i)
    end
end

function populate_all!(grid::BaselineGrid2D, node_init, ctx_init)
    populate_all!(grid.nodes, node_init)
    populate_all!(grid.node_ctx, ctx_init)
end

function init_forward_buffers!(grid::BaselineGrid2D{Node}, msg::Vector{<:AbstractMsg}) where {Node <: AbstractNode}
    for i in eachindex(grid.sum_buf1)
        if !isdefined(grid.sum_buf1, i) || is_samebuf(grid.sum_buf1[i], msg[1])
            grid.sum_buf1[i] = similar(msg[1])
        end
    end
    for i in eachindex(grid.sum_buf2)
        if !isdefined(grid.sum_buf2, i) || is_samebuf(grid.sum_buf2[i], msg[1])
            grid.sum_buf2[i] = similar(msg[1])
        end
    end
end

function forward(grid::BaselineGrid2D{Node}, msg::Vector{<:AbstractMsg}) where {Node <: AbstractNode}
    grid.sum_buf1[2:end-1] .= msg
    (d, l) = size(grid.nodes)
    for j in 1:l
        for i in 1:d
            y = forward(grid.nodes[i, j], grid.node_ctx[i, j], grid.sum_buf1[i+1])
            for k in grid.neighbour_offsets
                grid.sum_buf2[i+k+1] += y
            end
        end
        (grid.sum_buf1, grid.sum_buf2) = (grid.sum_buf2, grid.sum_buf1)
        zero!(grid.sum_buf2)
    end
    grid.sum_buf1[2:end-1]
end
