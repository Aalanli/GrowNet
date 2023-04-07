using Flux
using Flux: @adjoint

struct Node
    w::Matrix{Float32}
    b::Vector{Float32}
    dw::Matrix{Float32}
    db::Vector{Float32}
end

Node(dim::Int) = Node(randn(dim, dim), randn(dim), zeros(dim, dim), zeros(dim))

function relu(x)
    max(x, 0)
end

# x has shape [dim, batch]
function (node::Node)(x::Matrix{Float32})
    normx = Flux.normalise(x, dims=1)
    ty = relu.(node.w * normx .+ node.b)
    ty
end

function forward(node::Node, x)
    node(x)
end

@adjoint function forward(node::Node, x::Matrix{Float32})
    y, back = Flux.pullback((node, x) -> node(x), node, x)
    function modback(dy)
        ds = back(dy)
        dx = ds[2]
        dself = ds[1]
        node.dw .+= dself.w
        node.db .+= dself.b
        (nothing, dx)
    end
    y, modback
end

node = Node(4)

x = rand(Float32, 4, 16)
y, back = Flux.pullback(x -> forward(node, x), x)

grad = rand(Float32, 4, 16)

back(grad)
node.db

