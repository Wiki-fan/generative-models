import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data(BATCH_SIZE=100):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    # Download our data
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Use greyscale values as sampling probabilities to get back to [0,1]
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])

    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    # MNIST has no official train dataset so use last 10000 as validation
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()

    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    # Create data loaders
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)

    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter, val_iter, test_iter


class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """

    def __init__(self, image_shape, z_dim):
        super().__init__()

        self.__dict__.update(locals())

        hidden_dim = 400
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.generate = nn.Linear(hidden_dim, np.prod(image_shape))

    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = torch.sigmoid(self.generate(activated))
        generation = generation.view((x.shape[0],) + self.image_shape)
        return generation


class Discriminator(nn.Module):
    """ Discriminator. Input is an image (real or generated), output is P(generated).
    """

    def __init__(self, image_shape, output_dim):
        super().__init__()

        self.__dict__.update(locals())

        hidden_dim = 400
        self.linear = nn.Linear(np.prod(image_shape), hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        activated = F.relu(self.linear(x))
        discrimination = self.discriminate(activated)
        return discrimination
