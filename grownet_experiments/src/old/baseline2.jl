using Statistics
using Flux, Flux.Optimise
using MLDatasets
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition
#using CUDA


train_x, train_y = MLDatasets.MNIST.traindata(Float32)
train_x = repeat(train_x, outer = [1, 1, 1, 3])
train_x = permutedims(train_x, (1, 2, 4, 3))
labels = onehotbatch(train_y, 0:9)

# using Plots
# image(x) = colorview(RGB, permutedims(x, (3, 2, 1)))
# plot(image(train_x[:,:,:,rand(1:end)]))


train = ([(train_x[:,:,:,i], labels[:,i]) for i in partition(1:59000, 1000)]) #|> gpu
valset = 59001:60000
valX = train_x[:,:,:,valset] # |> gpu
valY = labels[:, valset] # |> gpu

m = Chain(
  Conv((5,5), 3=>16, relu),
  MaxPool((2,2)),
  Conv((5,5), 16=>8, relu),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(128, 120),
  Dense(120, 84),
  Dense(84, 10),
  softmax) # |> gpu

using Flux: crossentropy, Momentum

loss(x, y) = sum(crossentropy(m(x), y))
opt = Momentum(0.01)

accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))

epochs = 20

for epoch = 1:epochs
  for d in train
    gs = gradient(Flux.params(m)) do
      l = loss(d...)
    end
    update!(opt, Flux.params(m), gs)
  end
  @show accuracy(valX, valY)
end

test_x, test_y = CIFAR10.testdata(Float32)
test_labels = onehotbatch(test_y, 0:9)

test = ([(test_x[:,:,:,i], test_labels[:,i]) for i in partition(1:10000, 1000)])

ids = rand(1:10000, 5)
rand_test = test_x[:,:,:,ids]
rand_truth = test_y[ids]
m(rand_test)

accuracy(test[1]...)

class_correct = zeros(10)
class_total = zeros(10)
for i in 1:10
  preds = m(test[i][1])
  lab = test[i][2]
  for j = 1:1000
    pred_class = findmax(preds[:, j])[2]
    actual_class = findmax(lab[:, j])[2]
    if pred_class == actual_class
      class_correct[pred_class] += 1
    end
    class_total[actual_class] += 1
  end
end

class_correct ./ class_total
