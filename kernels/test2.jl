using Zygote

function normalize(ci)
    v = sum(ci .^ 2, dims=1) ./ size(ci)[1]
    v = sqrt.(v .+ 1e-6)
    
    (ci ./ v, v)
end

function my_dnormalize(grad, ci, v)
    temp = ci .* v .^ 3 ./ size(ci)[1]
    temp = sum(grad .* ci, dims=1) .* temp
    dy_dci = grad .* v .- temp
    return dy_dci
end


d_normalize(x, s) = gradient((x) -> sum(normalize(x)[1] .* s), x)


function main()
    ci = randn(16)
    (norm, v) = normalize(ci)
    gi = randn(16)
    my_dnorm = my_dnormalize(gi, ci, v)
    dnorm = d_normalize(ci, gi)[1]
    println(my_dnorm)
    println(dnorm)
end