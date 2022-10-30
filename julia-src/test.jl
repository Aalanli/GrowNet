using GLMakie, Colors
using LinearAlgebra
using GeometryBasics: Rect3f
GLMakie.activate!()
include("compute.jl")
using .compute

function example_plot()
    positions = vec([Point3f(i, j, k) for i = 1:7, j = 1:7, k = 1:7]) ## note 7 > 5 [factor in each i,j,k], whichs is misleading
    
    fig, ax, obj = meshscatter(positions;
        marker = Rect3f(Vec3f(-5.5), Vec3f(3.8)),
        transparency = true,
        color = [RGBA(positions[i]./7.0..., 0.5) for i in eachindex(positions)],
        figure = (;
            resolution = (1200, 800))
    )
    fig
    
    ps = [Point3f(x, y, z) for x = -3:1:3 for y = -3:1:3 for z = -3:1:3]
    ns = map(p -> 0.1 * rand() * Vec3f(p[2], p[3], p[1]), ps)
    lengths = norm.(ns)
    flowField(x, y, z) = Point(-y + x * (-1 + x^2 + y^2)^2, x + y * (-1 + x^2 + y^2)^2,
        z + x * (y - z^2))
    #fig = Figure(resolution=(1200, 800), fontsize=26)
    #axs = [Axis3(fig[1, i]; aspect=(1, 1, 1), perspectiveness=0.5) for i = 1:2]
    arrows!(ax, ps, ns, color=lengths, arrowsize=Vec3f(0.2, 0.2, 0.3),
        linewidth=0.1)
    
    fig
end

struct Test
    a::Int32
    b::Int64
    c::Int64
end

mutable struct TestMut
    a::Int32
    b::Int64
    c::Int64
end

struct C
    a::Vector{Int32}
end