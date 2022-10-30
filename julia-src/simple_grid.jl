using Statistics
using LinearAlgebra

struct Relu end

function forward(::Type{Relu}, x)
    max(x, 0)
end

function backward(::Type{Relu}, g, x)
    x > 0 ? 1 : 0 * g
end

mutable struct Grid{T<:AbstractFloat, ActFn}
    w::Array{T, 4} # [dim, dim, D, L]
    b::Array{T, 3} # [dim, D, L]
    dw::Array{T, 4}
    db::Array{T, 3}
    sum_buf1::Array{T, 3} # [dim, batch, D + 2]
    sum_buf2::Array{T, 3} # [dim, batch, D + 2]
    grid_size::NTuple{3, Int64} # [D, L]
end

function Grid(::Type{T}, ::Type{ActFn}, dimensions::NTuple{3, Int64}) where {T, ActFn}
    (dim, D, L) = dimensions
    Grid{T, ActFn}(
        randn(T, (dim, dim, D, L)),
        zeros(T, (dim, D, L)),
        zeros(T, (dim, dim, D, L)),
        zeros(T, (dim, D, L)),
        zeros(T, (1,1,1)),
        zeros(T, (1,1,1)),
        (dim, D, L)
    )
end


function normalize!(a::T) where {T<:AbstractMatrix}
    (d, l) = size(a)
    @inbounds @simd for i in 1:l
        mu = sum(@view(a[:, i])) / d
        sd = zero(typeof(mu))
        for j in 1:d
            sd += (a[j, i] - mu) ^ 2
        end
        sd = sqrt(sd / (d - 1))
        a[:, i] .= (@view(a[:, i]) .- mu) ./ (sd + 1e-6)
    end
end

function forward(grid::Grid{T, ActFn}, x::Array{T, 3}) where {T <: AbstractFloat, ActFn}
    if size(x) != size(grid.sum_buf1)
        grid.sum_buf1 = zeros(T, (size(x)[1:end-1]..., grid.grid_size[2] + 2))
        grid.sum_buf2 = zeros(T, (size(x)[1:end-1]..., grid.grid_size[2] + 2))
    end

    forward_kernel!(grid, x)

    grid.sum_buf1[:, :, 2:end-1]
end

function forward_kernel!(grid::Grid{T, ActFn}, x::Array{T, 3}) where {T <: AbstractFloat, ActFn}
    (dim, d, l) = grid.grid_size
    grid.sum_buf1[:, :, 2:end-1] .= x
    @inbounds for i in 1:l
        for j in 1:d
            mul!(@view(grid.sum_buf2[:, :, j+1]), @view(grid.w[:, :, j, i]), @view(grid.sum_buf1[:, :, j+1]))
            grid.sum_buf2[:, :, j+1] .= forward.(ActFn, @view(grid.b[:, j, i]) .+ @view(grid.sum_buf2[:, :, j+1]))
        end
        
        #(grid.sum_buf1, grid.sum_buf2) = (grid.sum_buf2, grid.sum_buf1)
        grid.sum_buf1 .= 0
        for j in 2:d+1
            for k in (-1, 0, 1)
                grid.sum_buf1[:, :, j] .= @view(grid.sum_buf1[:, :, j]) .+ @view(grid.sum_buf2[:, :, j+k])
            end
            normalize!(@view(grid.sum_buf1[:, :, j]))
        end

    end
end


function backward(grid::Grid{T, ActFn}, x::Array{T, 3}) where {T <: AbstractFloat, ActFn}
    if size(x) != size(grid.sum_buf1)
        grid.sum_buf1 = zeros(T, (size(x)[1:end-1]..., grid.grid_size[2] + 2))
        grid.sum_buf2 = zeros(T, (size(x)[1:end-1]..., grid.grid_size[2] + 2))
    else
        grid.sum_buf1 .= 0
        grid.sum_buf2 .= 0
    end

    backward_kernel!(grid, x)

    grid.sum_buf1[:, :, 2:end-1]
end

function backward_kernel!(grid::Grid{T, ActFn}, x::Array{T, 3}) where {T <: AbstractFloat, ActFn}
    (dim, d, l) = grid.grid_size
    grid.sum_buf1[:, :, 2:end-1] .= x
    for i in 1:l
        for j in 1:d
            mul!(@view(grid.sum_buf2[:, :, j+1]), @view(grid.w[:, :, j, i]), @view(grid.sum_buf1[:, :, j+1]))
            grid.sum_buf2[:, :, j+1] .= forward.(ActFn, @view(grid.b[:, j, i]) .+ @view(grid.sum_buf2[:, :, j+1]))
        end
        
        #(grid.sum_buf1, grid.sum_buf2) = (grid.sum_buf2, grid.sum_buf1)
        grid.sum_buf1 .= 0
        for j in 2:d+1
            for k in (-1, 0, 1)
                grid.sum_buf1[:, :, j] .= @view(grid.sum_buf1[:, :, j]) .+ @view(grid.sum_buf2[:, :, j+k])
            end
            normalize!(@view(grid.sum_buf1[:, :, j]))
        end

    end
end

function test()
    grid = Grid(Float32, Relu, (4, 64, 128))
    x = randn(Float32, (4, 2, 64))
    y = forward(grid, x)

    println(mean(y))
    println(maximum(y), " ", minimum(y))
    println(std(y))

    @time forward_kernel!(grid, x)
end

test()