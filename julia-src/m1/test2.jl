using Zygote

using Statistics

function normalize(v)
    mu = sum(v, dims=1) ./ size(v)[1]
    sd = Statistics.std(v)
    (v .- mu) ./ (sd .+ 1e-6)
end

d_normalize(x) = gradient((x) -> sum(normalize(x)), x)

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

function normalize2(a)
    k = deepcopy(a)
    normalize!(k)
    k
end

function d_normalize!(grad, x, dl_dx)
    d = size(a)[1]

    mu = sum(x) / d
    sd = Statistics.std(x)

    grad_sum = 0
    grad_dot = 0
    for j in 1:d
        grad_sum += grad[j]
        grad_dot += grad[j] * x[j]
    end
    rem1 = 1 / (sd + 1e-6)
    rem2 = 2 * mu / (d - 1) * sd
    grad_sum /= d
    grad_dot *= rem1

    for j in 1:d
        dl_dx[j] = rem1 * (grad[j] - grad_sum + grad_dot * (rem2 - (x[j] - mu)) / (sd * (d - 1)))
    end
end

function d_normalize2(x)
    x = deepcopy(x)
    buf = similar(x)
    grad = ones(eltype(x), size(x))
    d_normalize!(grad, x, buf)
    buf
end

d_std(x) = gradient(Statistics.std, x)

function d_std2(x)
    n = size(x)[1]
    mu = sum(x) / n
    sd = Statistics.std(x, mean=mu)
    c1 = -sum(x .- mu) / (sd * (n - 1) * n)
    dx = (x .- mu) ./ (sd * (n - 1)) .+ c1
    dx
end