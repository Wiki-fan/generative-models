""" (WGAN) https://arxiv.org/abs/1701.07875
Wasserstein GAN

The output of WGAN's D is unbounded unless passed through an activation
function. In this implementation, we include a sigmoid activation function
as this empirically improves visualizations for binary MNIST.

WGAN utilizes the Wasserstein distance to produce a value function which has
better theoretical properties than the vanilla GAN. In particular, the authors
prove that there exist distributions for which Jenson-Shannon, Kullback-Leibler,
Reverse Kullback Leibler, and Total Variaton distance metrics where Wasserstein
does. Furthermore, the Wasserstein distance has guarantees of continuity and
differentiability in neural network settings where the previously mentioned
distributions may not. Lastly, they show that that every distribution that
converges under KL, reverse-KL, TV, and JS divergences also converges under the
Wasserstein divergence and that a small Wasserstein distance corresponds to a
small difference in distributions. The downside is that Wasserstein distance
cannot be tractably computed directly. But if we make sure the discriminator
(aka Critic because it is not actually classifying) lies in the space of
1-Lipschitz functions, we can use that to approximate it instead. We crudely
enforce this via a weight clamping parameter C.

Note that this implementation uses RMSprop optimizer instead of Adam, as per
the original paper.
"""

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt
import numpy as np

from itertools import product
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class WGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class WGANDiscriminator(Discriminator):
            """ Critic (not trained to classify). Input is an image (real or generated),
                output is the approximate Wasserstein Distance between z~P(G(z)) and real.
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = WGANDiscriminator(image_shape, output_dim)


class WGANTrainer(TrainerBase):
    """ Object to hold data iterators, train a GAN variant
    """

    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.viz = viz
        self.num_epochs = 0

    def train(self, num_epochs, G_lr=5e-5, D_lr=5e-5, D_steps=5, clip=0.01):
        """ Train a Wasserstein GAN

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's RMProp optimizer
            D_lr: float, learning rate for discriminator's RMSProp optimizer
            D_steps: int, ratio for how often to train D compared to G
            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz
        """
        # Initialize optimizers
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                         if p.requires_grad], lr=G_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                         if p.requires_grad], lr=D_lr)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))

        # Begin training
        for epoch in tqdm(range(1, num_epochs + 1)):

            self.model.train()
            G_losses, D_losses = [], []

            for _ in range(epoch_steps):

                D_step_loss = []

                for _ in range(D_steps):
                    # Reshape images
                    images = self.process_batch(self.train_iter)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                    # Clamp weights (crudely enforces K-Lipschitz)
                    self.clip_D_weights(clip)

                # We report D_loss in this way so that G_loss and D_loss have
                # the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the generator to (roughly) minimize the approximated
                # Wasserstein distance
                G_loss = self.train_G(images)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)

            # Progress logging
            print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                  % (epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1

            # Visualize generator progress
            if self.viz:
                self.generate_images(epoch)
                plt.show()

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))]
        """
        # Sample from the generator
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Score real, generated images
        DX_score = self.model.D(images)  # D(x), "real"
        DG_score = self.model.D(G_output)  # D(G(x')), "fake"

        # Compute WGAN loss for D
        D_loss = -1 * (torch.mean(DX_score)) + torch.mean(DG_score)

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute WGAN loss for G
        G_loss = -1 * (torch.mean(DG_score))

        return G_loss

    def clip_D_weights(self, clip):
        for parameter in self.model.D.parameters():
            parameter.data.clamp_(-clip, clip)


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = WGAN(Generator, Discriminator, image_shape=(28, 28),
                 z_dim=20)

    # Init trainer
    trainer = WGANTrainer(model=model,
                          train_iter=train_iter,
                          val_iter=val_iter,
                          test_iter=test_iter,
                          viz=False)

    # Train
    trainer.train(num_epochs=100,
                  G_lr=5e-5,
                  D_lr=5e-5,
                  D_steps=5,
                  clip=0.01)
