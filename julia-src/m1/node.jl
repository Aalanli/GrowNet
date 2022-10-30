include("compute.jl")

import Base.+

struct Relu <: ParametricFn end

function forward(::Relu, x)
    max(x, 0)
end

function backward(::Relu, g, x)
    x > 0 ? 1 : 0 * g
end

struct LinearNode{T<:AbstractFloat, ActFn} <: AbstractNode
    w::Matrix{T}
    b::Vector{T}
    act::ActFn
end

mutable struct LinearNodeCtx{T<:AbstractFloat, ActFn} <: AbstractCtx
    x::Matrix{T}
    y::Matrix{T}
    dw::Matrix{T}
    db::Vector{T}
    act::ActFn
end

struct LinearNodeMsg{T<:AbstractFloat} <: AbstractMsg
    x::Matrix{T}
end

is_samebuf(a::LinearNodeMsg, b::LinearNodeMsg) = size(a.x) == size(b.x)
similar(a::LinearNodeMsg) = LinearNodeMsg(similar(a.x))

function LinearNode(::Type{T}, ::Type{ActFn}, dim::Int64) where {T<:AbstractFloat, ActFn}
    LinearNode{T, ActFn}(
        randn(dim, dim),
        zeros(dim),
        ActFn()
    )
end

function LinearNodeCtx(::Type{T}, ::Type{ActFn}, dim::Int64) where {T<:AbstractFloat, ActFn}
    LinearNodeCtx{T, ActFn}(
        zeros(1,1),
        zeros(1,1),
        zeros(dim, dim),
        zeros(dim),
        ActFn()
    )
end

ctx_type(::Type{ LinearNode{T, Act}}) where {T<:AbstractFloat, Act} = LinearNodeCtx{T, Act}
msg_type(::Type{ LinearNode{T, Act}}) where {T<:AbstractFloat, Act} = LinearNodeMsg{T}
grad_type(::Type{LinearNode{T, Act}}) where {T<:AbstractFloat, Act} = LinearNodeMsg{T}

function forward(node::LinearNode, ctx::LinearNodeCtx, msg::LinearNodeMsg)
    ctx.x = msg.x
    ctx.y = node.w * msg.x .+ node.b
    ctx.y ./= sqrt(size(node.w)[1])
    out = similar(ctx.y)
    for i in eachindex(out)
        out[i] = forward(node.act, ctx.y[i])
    end
    LinearNodeMsg(out)
end

function backward(node::LinearNode, ctx::LinearNodeCtx, grad::LinearNodeMsg)
    grad = backward.(node.act, grad, ctx.y)
    ctx.db .+= sum(grad, dims = 2)
    ctx.dw .+= grad * transpose(node.x)
    transpose(node.w) * grad
end

param_grads(node::LinearNode, ctx::LinearNodeCtx) = [(node.w, ctx.dw), (node.b, ctx.db)]

zero!(a::LinearNodeMsg) = a.x .= 0

(+)(a::LinearNodeMsg, b::LinearNodeMsg) = LinearNodeMsg(a.x + b.x)