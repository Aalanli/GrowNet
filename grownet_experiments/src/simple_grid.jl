using Flux
using Flux: @adjoint
using Statistics

struct Node
    w::Matrix{Float32}
    b::Vector{Float32}
end

Node(dim::Int) = Node(randn(dim, dim), randn(dim))

function leaky_relu(x)
    x > 0 ? x : x * 0.1
end

# x has shape [dim, batch]
function (node::Node)(x::Matrix{Float32})
    μ = mean(x, dims=1)
    sd = std(x, dims=1, mean=μ, corrected=false)
    normx = @. (x - μ) / (sd + 1)
    # println(normx)
    ty = leaky_relu.(node.w * normx .+ node.b)
    ty
end

@Flux.functor Node

# function forward(node::Node, x)
#     node(x)
# end

# @adjoint function forward(node::Node, x::Matrix{Float32})
#     y, back = Flux.pullback((node, x) -> node(x), node, x)
#     function modback(dy)
#         ds = back(dy)
#         dx = ds[2]
#         dself = ds[1]
#         node.dw .+= dself.w
#         node.db .+= dself.b
#         (nothing, dx)
#     end
#     y, modback
# end


struct Grid
    nodes::Array{Node, 3}
    dmodel::Int
    xyz::NTuple{3, Int}
end

@Flux.functor Grid

function Grid(dmodel::Int, x, y, z)
    nodes = Array{Node, 3}(undef, x, y, z)
    for i = 1:x, j = 1:y, k = 1:z
        nodes[i, j, k] = Node(dmodel)
    end
    Grid(nodes, dmodel, (z, y, z))
end

function apply_layer(nodes::AbstractArray{Node}, y::Array{Float32, 4})
    (dmodel, b, h, w) = size(y)
    store = Flux.Zygote.Buffer(zeros(Float32, dmodel, b, h, w))
    for ih = 1:h, iw = 1:w
        store[:, :, ih, iw] = nodes[ih, iw](y[:, :, ih, iw])
    end
    store = copy(store)
    store
end

function test_grad_applylayer()
    grid = Grid(4, 5, 5, 5)
    matnodes = @view grid.nodes[:, :, 1]
    x = rand(Float32, 4, 1, 5, 5)
    apply_layer(matnodes, x)
    y, back = Flux.pullback((matnodes, x) -> apply_layer(matnodes, x), matnodes, x)
    dx = back(y)
    println(dx[1])
    any(isnan.(dx[2]))
end

test_grad_applylayer()

function (grid::Grid)(im::Array{Float32, 4}) # [h, w, dmodel, b]
    y = permutedims(im, [3, 4, 1, 2])
    z = grid.xyz[3]
    for iz = 1:z
        store = apply_layer(@view(grid.nodes[:, :, iz]), y)
        y = Flux.NNlib.meanpool(store, (3, 3), pad=1, stride=1)
    end
    y
end


function make_model(dmodel, x, y, z, classes)
    Chain(
        Conv((3, 3), 3 => dmodel, relu, pad=1), # [x, y, 3, b] => [x, y, dmodel, b]
        # Grid(dmodel, x, y, z), # [x, y, dmodel, b] => [dmodel, b, x, y]
        im -> permutedims(im, [3, 4, 1, 2]),
        im -> reshape(im, x * y * dmodel, :),
        Dense(x * y * dmodel, classes),
        softmax
    )
end

using Flux.Optimise
using MLDatasets
using Images.ImageCore
using Base.Iterators: partition
using Flux: onehotbatch, onecold
#using CUDA


train_x, train_y = MLDatasets.MNIST.traindata(Float32)
train_x = repeat(train_x, outer = [1, 1, 1, 3])
train_x = permutedims(train_x, (1, 2, 4, 3))
labels = onehotbatch(train_y, 0:9)

train = ([(train_x[:,:,:,i], labels[:,i]) for i in partition(1:59000, 32)]) #|> gpu
valset = 59001:60000
valX = train_x[:,:,:,valset] # |> gpu
valY = labels[:, valset] # |> gpu

m = make_model(4, 28, 28, 5, 10)

using Flux: crossentropy, Momentum

loss(x, y) = sum(crossentropy(m(x), y))
opt = Momentum(0.01)

accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))

epochs = 20

using Flux.Zygote: @nograd
function step!(d)
    gs = gradient(Flux.params(m)) do
        l = loss(d...)
        println(l)
        l
    end
    update!(opt, Flux.params(m), gs)
end

for epoch = 1:epochs
  for d in train
    step!(d)
  end
  @show accuracy(valX, valY)
end