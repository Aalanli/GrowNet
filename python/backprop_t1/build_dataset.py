# %%
# abstracts away the dataset loading and unloading
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

CLASS_REPR = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def collate_fn(data):
    """pass the collation function to the Dataloader to batch images
       into a single tensor"""
    images = [d[0] for d in data]
    labels = [d[1] for d in data]
    return torch.stack(images, dim=0), torch.tensor(labels)

def get_dataset(batch_size, augment=True):
    # some dataset transforms, to introduce some invariances to the model
    # such as scale invariance, rotation invariance, etc.
    # finally noramlize the image for better training dynamics
    transform = [AutoAugment(AutoAugmentPolicy.CIFAR10)] if augment else []
    transform = transforms.Compose(
     transform + [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # we don't want augmentations on the val set
    transform_val = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, collate_fn=collate_fn)

    # do not train on the test dataset, to better gauge generalization error
    # instead of training error, test dataset should not include any training
    # images
    testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                        download=True, transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, collate_fn=collate_fn)

    return trainloader, testloader


# some unit testing to see that it all works
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    train_loader, test_loader = get_dataset(4, augment=False)
    images, labels = next(iter(train_loader))
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()
    print("class:", CLASS_REPR[labels[0]])
