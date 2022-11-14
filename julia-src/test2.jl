function transpose_test(a::AbstractArray, i, j)
    st_i = stride(a, i)
    st_j = stride(a, j)
    dims = size(a)

    for i = 0:dims[i]-1
        for j = i:dims[j]-1
            p = a[i * st_i + j * st_j + 1]
            a[i * st_i + j * st_j + 1] = a[i * st_j + j * st_i + 1]
            a[i * st_j + j * st_i + 1] = p
        end
    end
end

function permute_test(a, dims)
    for i = dims[1:end-1]
        if dims[i] != i
            transpose_test(a, i, dims[i])
        end
    end
end

function test_permute()
    a = randn(5, 5, 1)
    b = deepcopy(a)

    a = permutedims(a, [2, 1, 3])
    #permute_test(b, [3, 2, 1])
    transpose_test(b, 1, 2)
    isapprox(a, b)
end

function test_transpose()
    a = randn(3, 7)
    b = deepcopy(a)
    a = transpose(a)
    transpose_test(b, 1, 2)
    b = reshape(b, (7, 3))

    isapprox(a, b)
end