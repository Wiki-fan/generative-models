import torch
import torch.nn as nn


def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()


def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class GAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)
        self.D = Discriminator(image_shape, output_dim)
