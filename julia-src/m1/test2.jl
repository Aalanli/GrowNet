include("compute.jl")
include("grid.jl")
include("node.jl")

function test_grid()
    grid = BaselineGrid2D((128, 128), LinearNode{Float32, Relu})
    populate_all!(grid, (_) -> LinearNode(Float32, Relu, 4), (_) -> LinearNodeCtx(Float32, Relu, 4))
    x = [LinearNodeMsg(randn(Float32, 4, 5)) for i in 1:128]
    init_forward_buffers!(grid, x)
    @time forward(grid, x)
end

test_grid()