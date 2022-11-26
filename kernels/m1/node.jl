struct Relu end

function forward(::Type{Relu}, x)
    max(x, 0)
end

function backward(::Type{Relu}, g, x)
    x > 0 ? 1 : 0 * g
end

function normalize!(a::T, mu_v, sd_v) where {T<:AbstractMatrix}
    (d, l) = size(a)
    @inbounds @simd for i in 1:l
        mu = sum(@view(a[:, i])) / d
        sd = zero(typeof(mu))
        for j in 1:d
            sd += (a[j, i] - mu) ^ 2
        end
        sd = sqrt(sd / (d - 1))
        a[:, i] .= (@view(a[:, i]) .- mu) ./ (sd + 1e-6)
        mu_v[i] = mu
        sd_v[i] = sd
    end
end

function d_normalize!(grad, x, dl_dx, mu_v, sd_v)
    (d, l) = size(a)
    for i in 1:l
        mu = mu_v[i]
        sd = sd_v[i]

        grad_sum = 0
        grad_dot = 0
        for j in 1:d
            grad_sum += grad[j, i]
            grad_dot += grad[j, i] * x[j, i]
        end
        rem1 = 1 / (sd + 1e-6)
        rem2 = 2 * mu / (d - 1) * sd
        grad_sum /= d
        grad_dot *= rem1

        for j in 1:d
            dl_dx[j, i] = rem1 * (grad[j, i] - grad_sum + grad_dot * (rem2 - (x[j, i] - mu)) / (sd * (d - 1)))
        end
    end
end

struct LinearNode{T<:AbstractFloat, ActFn}
    w::Matrix{T}
    b::Vector{T}
    act::ActFn
end

mutable struct LinearNodeCtx{T<:AbstractFloat, ActFn}
    x::Matrix{T}
    y::Matrix{T}
    z::Matrix{T}
    mu::Vector{T}
    sd::Vector{T}

    dl_dy::Matrix{T}
    dy_dn::Matrix{T}
    dw::Matrix{T}
    db::Vector{T}
end

mutable struct GeneralCtx{T<:AbstractFloat, N}
    arr::Vector{T}
    reserved::Int64
    shapes::Vector{NTuple{N, Int64}}
    offsets::Vector{Tuple{Int64, Int64}}
end

function GeneralCtx(::Type{T}, ::Val{N}) where {T<:AbstractFloat, N}
    GeneralCtx{T, N}(
        zeros(T, 1),
        1,
        Vector{NTuple{N, Int64}}(undef, 0),
        Vector{Tuple{Int64, Int64}}(undef, 0)
    )
end

struct CtxView
    i::Int64
end

function reserve!(ctx::GeneralCtx{<:AbstractFloat, N}, shape::NTuple{N, Int64}) where {N}
    alloc = prod(shape)
    if ctx.reserved + alloc > length(ctx.arr)
        resize!(ctx.arr, alloc + ctx.reserved)
    end
    ctx.arr[ctx.reserved:ctx.reserved + alloc] .= 0
    push!(ctx.offsets, (ctx.reserved, alloc))
    push!(ctx.shapes, shape)
    ctx.reserved += alloc
    CtxView(length(ctx.offsets))
end

function get_view(ctx::GeneralCtx, view_id::CtxView)
    i = view_id.i
    shape = ctx.shapes[i]
    (offsets, size) = ctx.offsets[i]
    slice = @view(ctx.arr[offsets:offsets + size-1])
    slice = reshape(slice, shape)
end
    
function clear!(ctx::GeneralCtx)
    ctx.reserved = 1
    empty!(ctx.offsets)
    empty!(ctx.shapes)
    nothing
end

function LinearNode(::Type{T}, ::Type{ActFn}, dim::Int64) where {T<:AbstractFloat, ActFn}
    LinearNode{T, ActFn}(
        randn(dim, dim),
        zeros(dim),
        ActFn()
    )
end

function LinearNodeCtx(::Type{T}, ::Type{ActFn}, dim::Int64) where {T<:AbstractFloat, ActFn}
    LinearNodeCtx{T, ActFn}(
        zeros(dim,1),
        zeros(dim,1),
        zeros(dim,1),
        zeros(1),
        zeros(1),
        zeros(dim, dim),
        zeros(dim),
        ActFn()
    )
end

ctx_type(::Type{LinearNode{T, Act}}) where {T<:AbstractFloat, Act} = LinearNodeCtx{T, Act}

function forward(node::LinearNode, ctx::LinearNodeCtx)
    ctx.y .= node.w * ctx.x .+ node.b
    for i in eachindex(ctx.y)
        ctx.y[i] = forward(node.ActFn, ctx.y[i])
    end
    ctx.z .= ctx.y
    normalize!(ctx.z, ctx.mu, ctx.sd)
    ctx.z
end

function backward(node::LinearNode, ctx::LinearNodeCtx)
    d_normalize!(ctx.dl_dy, ctx.y, ctx.dy_dn, ctx.mu, ctx.sd)
    ctx.dy_dn .= backward.(node.ActFn, ctx.dy_dn, ctx.y)
    ctx.db .+= sum(ctx.dy_dn, dims = 2)
    ctx.dw .+= ctx.dy_dn * transpose(ctx.x)
    transpose(node.w) * ctx.dy_dn
end
