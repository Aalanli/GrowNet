# %%
import os
os.chdir('python')

# %%
from itertools import cycle

import torch
from torch import nn, optim
from matplotlib import pyplot as plt
import numpy as np

from backprop_t1.convnext import convnext_small
from backprop_t1.build_dataset import get_dataset
from backprop_t1.stylegan2 import Generator

# parameters for training
epochs       = 10
loss_file    = 'losses.txt'
plot_file    = 'plots.png'

batch_size   = 32
lr           = 1e-4

gen_z_dim    = 256
gen_w_dim    = 256
gen_out_res  = 32
gen_in_res   = 8
feat_map_no  = 1

# I want to fine-tune the model, since this results in higher
# total accuracy and is more energy efficient to train
embedding = convnext_small(pretrained=True, num_classes=10).cuda()
feat_maps, _ = embedding(torch.randn([1, 3, 32, 32]).cuda())
dims = [3] + [xs.shape[1] for (i, xs) in feat_maps.items()]
is_last = feat_map_no == 1
generator = Generator(gen_z_dim, gen_w_dim, gen_in_res, gen_out_res, dims[feat_map_no], dims[feat_map_no-1], is_last).cuda()


# %%
# building the dataset, see build_dataset for more details
train_loader, test_loader = get_dataset(batch_size, augment=True)
# make the test_loader be an infinite cycle, so StopIteration never occurs
test_loader_t = cycle(iter(test_loader))

# construct the losses, in this case, the negative log-likehood loss
# which is used for classification tasks. It is the log of the softmax
# of the logits multiplied by the label. I don't want to implement label
# smoothing here, since its over-kill
criterion = nn.MSELoss()

# Use the adam optimizer for lesser hyper-parameters
optimizer = optim.Adam(generator.parameters(), lr=lr)

# save the loss ever log_steps
log_step     = 500
val_losses   = []
train_losses = []
loss_msg     = []
# start the training loop
# in a more complex project, I would usually separate this
# in a separate function or class

total_steps = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_val_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # set the model to the train stage, since sometimes
        # dropout has different behaviors
        generator.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # move data to gpu for acceleration
        img = inputs.cuda()
        feat_maps = [img]
        with torch.no_grad():
            emb, _ = embedding(img)
            feat_maps = feat_maps + [xs for (_, xs) in emb.items()]

        x, y = feat_maps[feat_map_no], feat_maps[feat_map_no-1]
        latent = torch.randn([x.shape[0], gen_z_dim], device='cuda')
        xhat = generator(latent, x)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = criterion(xhat, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_step == (log_step - 1):    # print every log_step mini-batches
            train_msg = f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / log_step:.3f} \n'
            eval_msg = f'[{epoch + 1}, {i + 1:5d}] eval loss: {running_val_loss / log_step:.3f} \n'
            print(train_msg)
            print(eval_msg)

            loss_msg.append(train_msg)
            loss_msg.append(eval_msg)
            train_losses.append(running_loss / log_step)
            val_losses.append(running_val_loss / log_step)

            running_loss = 0.0
            running_val_loss = 0.0
        total_steps += 1


print('Finished Training')

# plot losses over time
plt.plot(np.array(train_losses), label='train-loss')
plt.plot(np.array(val_losses), label='val-loss')
plt.xlabel("steps")
plt.ylabel('mean-crossentropy-loss')
plt.title('mean-crossentropy-loss over steps')
plt.legend()
plt.savefig(plot_file)
plt.show()

# %%
h, _ = next(test_loader_t)
plt.imshow((h[0].permute(1, 2, 0) + 1) / 2)
plt.show()

with torch.no_grad():
    img = h.cuda()
    feat_maps = [img]
    emb, _ = embedding(img)
    feat_maps = feat_maps + [xs for (_, xs) in emb.items()]

    x, y = feat_maps[feat_map_no], feat_maps[feat_map_no-1]
    latent = torch.randn([x.shape[0], gen_z_dim], device='cuda')
    xhat = generator(latent, x)

    plt.imshow((xhat[0].permute(1, 2, 0) + 1).cpu() / 2)
    plt.show()
