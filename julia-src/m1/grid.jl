include("compute.jl")
import Base.similar

mutable struct BaselineGrid2D{
    N<:AbstractNode, Msg, GradMsg, Ctx<:AbstractCtx} <: AbstractGrid
    nodes::Array{N, 2}
    node_ctx::Array{Ctx, 2}
    sbuf::Array{Msg, 1}
    rbuf::Array{Msg, 1}
    neighbour_offsets::NTuple{3, Int64}
end

function BaselineGrid2D(dims::NTuple{2, Int}, ::Type{Node}) where {Node <: AbstractNode}
    offsets = (-1, 0, 1)
    nodes = Array{Node, 2}(undef, dims)
    node_ctx = Array{ctx_type(Node), 2}(undef, dims)

    sbuf = Array{msg_type(Node), 1}(undef, dims[1] + 2) # generic function to compute this
    rbuf = Array{msg_type(Node), 1}(undef, dims[1] + 2) # generic function to compute this

    BaselineGrid2D{Node, msg_type(Node), grad_type(Node), ctx_type(Node)}(
        nodes, 
        node_ctx,
        sbuf,
        rbuf,
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


function forward(grid::BaselineGrid2D{Node}, msg::Vector) where {Node <: AbstractNode}
    init_ctx!(grid.node_ctx, msg)
    init_buf!(grid.sbuf, msg)
    init_buf!(grid.rbuf, msg)
end

# should allocate no memory
function forward_kernel(grid::BaselineGrid2D{Node}, msg::Vector) where {Node <: AbstractNode}
    (d, l) = size(grid.nodes)
    for i in 1:d
        grid.sbuf[i+1] = msg[i]
    end

    for j in 1:l
        for i in 1:d
            forward(grid.nodes[i, j], grid.node_ctx[i, j], grid.sbuf[i+1])
        end
        pool!(grid.sbuf, grid.rbuf)
        grid.rbuf, grid.sbuf = grid.sbuf, grid.rbuf
    end
end
