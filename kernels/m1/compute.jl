import Base.+

# Abstract Container types for message passing
abstract type AbstractCtx end

# Abstract Function types
abstract type AbstractNode end


forward(m::AbstractNode, ctx::AbstractCtx, msg) =
    throw("forward method for $(typeof(m)), $(typeof(ctx)) and $(typeof(msg)) is not implemented")

backward(m::AbstractNode, ctx::AbstractCtx, grad) =
    throw("backward method for $(typeof(m)), $(typeof(ctx)) and $(typeof(grad)) is not implemented")

param_grads(m::AbstractNode, ctx::AbstractCtx) = 
    throw("no method found for computing parameter and gradient info")


# gets the type of the gradient message that the node receives
grad_type(::Type{T}) where {T <: AbstractNode} = 
    throw("cannot extract node msg type for type $T")

# gets the type of the context object that node requires
ctx_type(::Type{T}) where {T <: AbstractNode} = 
    throw("cannot extract node msg type for type $T")
