import Base.+

# Abstract Container types for message passing
abstract type AbstractMsg end
abstract type AbstractCtx end

# Abstract Function types
abstract type ParametricFn end
abstract type AbstractNode end
abstract type AbstractGrid end

forward(m::ParametricFn, ctx::AbstractCtx, msg::AbstractMsg) =
    throw("forward method for $(typeof(m)), $(typeof(ctx)) and $(typeof(msg)) is not implemented")

backward(m::ParametricFn, ctx::AbstractCtx, grad::AbstractMsg) =
    throw("backward method for $(typeof(m)), $(typeof(ctx)) and $(typeof(grad)) is not implemented")

param_grads(m::ParametricFn, ctx::AbstractCtx) = 
    throw("no method found for computing parameter and gradient info")

# gets the type of the message that the node receives
msg_type(::Type{T}) where {T <: AbstractNode} = 
    throw("cannot extract node msg type for type $T")

# gets the type of the gradient message that the node receives
grad_type(::Type{T}) where {T <: AbstractNode} = 
    throw("cannot extract node msg type for type $T")

# gets the type of the context object that node requires
ctx_type(::Type{T}) where {T <: AbstractNode} = 
    throw("cannot extract node msg type for type $T")

zero!(a::AbstractMsg) = 
    throw("zero! not implemented for type $(typeof(a))")

(+)(a::AbstractMsg, ::AbstractMsg) = 
    throw("+ not implemented for type $(typeof(a))")