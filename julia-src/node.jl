include("compute.jl")


struct LinearNode{T<:AbstractFloat, ActFn} <: AbstractNode
    w::Matrix{T}
    b::Vector{T}
    act::ActFn
end

struct LinearNodeCtx{T<:AbstractFloat, ActFn} <: AbstractCtx
    x::Matrix{T}
    y::Matrix{T}
    dw::Matrix{T}
    db::Vector{T}
    act::ActFn
end

struct LinearNodeMsg{T<:AbstractFloat} <: AbstractMsg
    x::Matrix{T}
end

ctx_type(::Type{LinearNode}) = LinearNodeCtx
msg_type(::Type{LinearNode}) = LinearNodeMsg
grad_type(::Type{LinearNode}) = LinearNodeMsg

function forward(node::LinearNode, ctx::LinearNodeCtx, msg::LinearNodeMsg) 
end

function backward(node::LinearNode, ctx::LinearNodeCtx, msg::LinearNodeMsg)
end
