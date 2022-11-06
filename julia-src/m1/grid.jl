
mutable struct BaselineGrid2D{Node, Ctx}
    nodes::Array{Node, 2}
    node_ctx::Array{Ctx, 2}
    neighbour_offsets::NTuple{3, Int64}
end

function BaselineGrid2D(dims::NTuple{2, Int}, ::Type{Node}) where {Node}
    offsets = (-1, 0, 1)
    nodes = Array{Node, 2}(undef, dims)
    node_ctx = Array{ctx_type(Node), 2}(undef, (dims[1]+1, dims[2]))

    BaselineGrid2D{Node, ctx_type(Node)}(
        nodes, 
        node_ctx,
        offsets
    )
end


function forward(grid::BaselineGrid2D{Node}, msg) where {Node}
    init_ctx!(grid.node_ctx, msg)
end

# should allocate no memory
function forward_kernel(grid::BaselineGrid2D{Node}, msg) where {Node}
    (d, l) = size(grid.nodes)
    for i in 1:d
        activate!(grid.node_ctx[i], msg[i])
    end

    for j in 1:l
        for i in 1:d
            y = forward(grid.nodes[i, j], grid.node_ctx[i, j])
            for k in grid.neighbour_offsets
                activate!(grid.node_ctx[i + k, j+1], y)
            end
        end
    end
    @view(grid.node_ctx[:, l])
end
