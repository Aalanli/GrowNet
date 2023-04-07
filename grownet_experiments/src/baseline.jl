using Flux, MLDatasets, CUDA
using Images, Augmentor, Parameters, IterTools

@with_kw struct Config
    batchsize::Int = 32
    lr::Float32 = 1f-3
    epochs::Int = 2
end

config = Config()

train_data = MLDatasets.CIFAR10(Float32, split=:train)
test_data = MLDatasets.CIFAR10(Float32, split=:test)

train_aug = FlipX(0.5) |> ShearX(-5:5) * ShearY(-5:5) |> Rotate(-15:15) |> 
            CropSize(32, 32) |> Zoom(0.9:0.1:1.2) |> SplitChannels() |>
            PermuteDims(3, 2, 1) |> ConvertEltype(Float32)

function collate((imgs, labels))
    imgs = imgs |> gpu
    labels = Flux.onehotbatch(labels .+ 1, 1:10) |> gpu
    imgs, labels
end

function collate((imgs, labels), aug)
    imgs_aug = Array{Float32}(undef, size(imgs))
    augmentbatch!(imgs_aug, MLDatasets.convert2image(MLDatasets.CIFAR10, imgs), aug)
    collate((imgs_aug, labels))
end

train_loader = imap(d -> collate(d, train_aug), Flux.Data.DataLoader(train_data, 
    batchsize=config.batchsize, shuffle=true))

test_loader = imap(collate, Flux.Data.DataLoader(test_data, 
    batchsize=config.batchsize, shuffle=false))

function conv_block(ch::Pair; kernel_size=3, stride=1, activation=relu)
    Chain(Conv((kernel_size, kernel_size), ch, pad=SamePad(), stride=stride, init=Flux.kaiming_normal),
        BatchNorm(ch.second, activation))
end

function basic_residual(ch::Pair)
    Chain(conv_block(ch),
        conv_block(ch.second => ch.second, activation=identity))
end


struct AddMerge
    gamma
    expand
end

Flux.@functor AddMerge

function AddMerge(ch::Pair)
    if ch.first == ch.second
        expand = identity
    else
        expand = conv_block(ch, kernel_size=1, activation=identity)
    end
    AddMerge([0.f0], expand)
end

(m::AddMerge)(x1, x2) = relu.(m.gamma .* x1 .+ m.expand(x2))


function residual_block(ch::Pair)
    residual = basic_residual(ch)
    SkipConnection(residual, AddMerge(ch))
end

function residual_body(in_channels, repetitions, downsamplings)
    layers = []
    res_channels = in_channels
    for (rep, stride) in zip(repetitions, downsamplings)
        if stride > 1
            push!(layers, MaxPool((stride, stride)))
        end
        for i = 1:rep
            push!(layers, residual_block(in_channels => res_channels))
            in_channels = res_channels
        end
        res_channels *= 2
    end
    Chain(layers...)
end

function stem(in_channels=3; channel_list=[32, 32, 64], stride=1)
    layers = []
    for channels in channel_list
        push!(layers, conv_block(in_channels => channels, stride=stride))
        in_channels = channels
        stride = 1
    end
    Chain(layers...)
end

function head(in_channels, classes, p_drop=0.0)
    Chain(GlobalMeanPool(),
          Flux.flatten,
          Dropout(p_drop),
          Dense(in_channels, classes))
end

function resnet(classes, repetitions, downsamplings; in_channels=3, p_drop=0.0)
    stem(in_channels, stride=downsamplings[1])
    Chain(stem(in_channels, stride=downsamplings[1]),
          residual_body(64, repetitions, downsamplings[1:end]),
          head(64 * 2 ^ (length(repetitions) - 1), classes, p_drop))
end

model = resnet(10, [2, 2, 2, 2], [1, 1, 2, 2, 2], p_drop=0.3) |> gpu

loss(x, y) = Flux.logitbinarycrossentropy(model(x), y)

a = rand(32, 32, 3, 2) |> gpu
y = rand(10, 2) |> gpu



# ps = Flux.params(model)

# opt = Flux.Optimiser(InvDecay(0.001), Flux.AdamW(config.lr, (0.9, 0.999), 1f-4))

# function accuracy(model, data)
#     m = Mean()
#     for (x, y) in data
#         fit!(m, Flux.onecold(cpu(model(x)), 1:10) .== Flux.onecold(cpu(y), 1:10))
#     end
#     value(m)
# end

# using Printf
# evalcb = Flux.throttle(20) do 
#     @printf "Val accuracy: %.3f\n" accuracy(model, test_loader)
# end


# Flux.@epochs config.epochs Flux.train!(loss, ps, train_loader, opt, cb=evalcb)