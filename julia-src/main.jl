using StaticArrays

struct GlobalParams{T<:AbstractFloat, Dim, ActFn}
    lr::T
    compute_dim::Int64
    grid_dim::NTuple{Dim, Int64}
    act::ActFn
end

struct Relu
end

function forward(::Relu, x)
    max.(x, 0)
end

function backward(::Relu, g, x)
    df = (i) -> i > 0 ? 1 : 0
    g .* df.(x)
end


mutable struct ComputeInstance{T<:AbstractFloat, ActFn}
    w::Matrix{T}
    b::Vector{T}
    dw::Matrix{T}
    db::Vector{T}
    act::ActFn
    x::Matrix{T}
    y::Matrix{T}
end

function ComputeInstance(::Type{T}, n::Int, ::Type{ActFn}) where {T <: AbstractFloat, ActFn}
    ComputeInstance{T, ActFn}(
        randn(n, n),
        randn(n),
        zeros(n, n),
        zeros(n),
        ActFn(),
        zeros(1,1),
        zeros(1,1)
        )    
end

function ComputeInstance(n::Int)
    ComputeInstance(Float32, n, Relu)
end

function forward(node::ComputeInstance, x::Matrix)
    node.x = x
    node.y = node.w * x .+ node.b
    forward(node.act, node.y)
end

function backward(node::ComputeInstance, grad::Matrix)
    grad = backward(node.act, grad, node.y)
    node.db .+= sum(grad, dims = 2)
    node.dw .+= grad * transpose(node.x)
    transpose(node.w) * grad
end

function apply_grad!(node::ComputeInstance, params::GlobalParams)
    node.w .-= node.dw .* params.lr
    node.b .-= node.db .* params.lr
end

function zero_grad!(node::ComputeInstance)
    node.dw .= 0
    node.db .= 0
end

mutable struct BaselineGrid2D{T<:AbstractFloat, ActFn}
    nodes::Array{ComputeInstance{T, ActFn}, 2}
    sum_buf1::Array{T, 3}
    sum_buf2::Array{T, 3}
    c_dim::Int64
    neighbour_offsets::NTuple{3, Int64}
end

function BaselineGrid2D(params::GlobalParams{T, 2, ActFn}) where {T <:AbstractFloat, ActFn}
    (a, b) = (params.grid_dim[1], params.grid_dim[2] + 2)
    arr = Array{ComputeInstance{T, ActFn}, 2}(undef, params.grid_dim)
    for i in eachindex(arr)
        arr[i] = ComputeInstance(T, params.compute_dim, ActFn)
    end
    offsets = (a - 1, a, a + 1)
    BaselineGrid2D{T, ActFn}(
        arr,
        zeros(1,1,1),
        zeros(1,1,1),
        params.compute_dim,
        offsets
    )
end

function set_batch_sz!(model::BaselineGrid2D, batch)
    a = size(model)[1]
    model.sum_buf1 = zeros(model.compute_dim, batch, a)
    model.sum_buf2 = zeros(model.compute_dim, batch, a)
end

function forward(model::BaselineGrid2D, x::Array{<:AbstractFloat, 3})
    if size(x)[2] != size(model.sum_buf)[2]
        set_batch_sz!(model, size(x)[2])
    end
    
    forward_kernel(model, x)
end

function forward_kernel(model::BaselineGrid2D, x::Array{<:AbstractFloat, 3})
    (d, l) = size(model.nodes)
    model.sum_buf1[:, :, 2:end-1] .= x
    for j in 1:l
        for i in 1:d
            y = forward(model.nodes[i, j], model.sum_buf[:, :, i+1])
            for k in model.neighbour_offsets
                model.sum_buf2[:, :, i+k+1] .+= y
            end
        end
        (model.sum_buf1, model.sum_buf2) = (model.sum_buf2, model.sum_buf1)
        model.sum_buf2 .= 0
    end
    model.sum_buf1
end


struct FMessage{T<:AbstractFloat, N}
    msg::SVector{N, T}
end

struct BMessage{T<:AbstractFloat, N}
    grad::SVector{N, T}
end



function test_compute_instance()
    compute = ComputeInstance(3)
    x = randn(3, 4)
    y = forward(compute, x)
    backward(compute, y)
end