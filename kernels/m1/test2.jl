using Zygote

using Statistics

function normalize(v)
    mu = sum(v, dims=1) ./ size(v)[1]
    sd = Statistics.std(v)
    (v .- mu) ./ (sd .+ 1e-6)
end

d_normalize(x, s) = gradient((x) -> sum(normalize(x) .* s), x)

function normalize!(a)
    d = size(a)[1]

    mu = sum(a) / d
    sd = zero(typeof(mu))
    for j in 1:d
        sd += (a[j] - mu) ^ 2
    end
    sd = sqrt(sd / (d - 1))
    a .= (a .- mu) ./ (sd + 1e-6)
end

function d_normalize!(grad, x)
    d = size(x)[1]

    mu = sum(x) / d
    sd = Statistics.std(x)

    dl_dx = grad ./ (sd + 1e-6)
    d_std = -sum(grad .* x) / (sd + 1e-6) ^ 2
    d_std = d_std .* d_std2(x)
    dl_dx .+= d_std
    
    s = 1 / (d * (sd + 1e-6))
    dx = -mu .* d_std2(x) / (sd + 1e-6) ^ 2
    dx .+= s
    dx .*= sum(grad)

    dl_dx .- dx
end


d_std(x) = gradient(Statistics.std, x)

function d_std2(x)
    n = size(x)[1]
    mu = sum(x) / n
    sd = Statistics.std(x, mean=mu)
    dx = (x .- mu) ./ (sd * (n - 1))
    dx
end

function d_std_test()
    a = randn(512)
    y1 = d_std2(a)
    y2, = d_std(a)
    isapprox(y1, y2)
end

function d_normalize_test()
    a = randn(512)
    grad = randn(512)
    y1 = d_normalize2(grad, a)
    y2, = d_normalize(a, grad)
    println("y1 $(y1), y2 $y2")
    isapprox(y1, y2)
end

function d_normalize2(grad, x)
    n = size(x)[1]
    mu = sum(x) / n
    sd = Statistics.std(x, mean=mu)

    norm = 1 / (sd + 1e-6)
    dotx = 0
    sum_grad = 0
    for i in 1:n
        sum_grad += grad[i]
        dotx += grad[i] * x[i]
    end

    dy_dx = @. norm * (grad - sum_grad / n + (x - mu) * (sum_grad * mu - dotx) * norm / ((n - 1) * sd))
    dy_dx
end

d_normalize_test()