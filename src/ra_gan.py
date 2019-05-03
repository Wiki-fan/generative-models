""" (RaGAN) https://arxiv.org/abs/1807.00734
Relativistic GAN

Relativistic GANs argue that the GAN generator should decrease the
discriminator's output probability that real data is real in addition to
increasing its output probability that fake data is real. By doing this, GANs
are claimed to be more stable and generate higher quality images.

Discriminator loss is changed such that the discriminator estimates the
probability that the given real data is more realistic than a randomly sampled
fake data. Generator loss is change such that real data is less likely to be
classified as real and fake data is more likely to be classified as real.

For computational efficiency, the discriminator estimates the probability that
the given real data is more realistic than fake data, on average. Otherwise,
the network would need to consider all combinations of real and fake data in the
minibatch. This would require O(m^2) instead of O(m), where m is batch size.

L(D) = -E[log( sigmoid(D(x) - E[D(G(z))]) )]
        - E[log(1 - sigmoid(D(G(z)) - E[D(x)]))]

L(G) = -E[log( sigmoid(D(G(z)) - E[D(x)]) )]
        - E[log(1 - sigmoid(D(x) - E[D(G(z))]))]

This implementation uses non-saturating (NS) GAN as a case study. The actual
modification proposed herein can be applied to any GAN in which the output of
the discriminator can be interpretted as a probability.
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


class RaNSGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_size, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_size, z_dim)

        class RaNSGANDiscriminator(Discriminator):
            """ Discriminator. Input is an image (real or generated),
                output is P(generated).
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = RaNSGANDiscriminator(image_size, output_dim)


class RaNSGANTrainer(TrainerBase):
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

    def train(self, num_epochs, G_lr=2e-4, D_lr=2e-4, D_steps=1):
        """ Train a relativistic non-saturating GAN

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
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

                    # Learn to discriminate between real and generated images
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                # So that G_loss and D_loss have the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Learn to generate images that fool the discriminator
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
            D_loss: non-saturing loss for discriminator,
            -E[log( sigmoid(D(x) - E[D(G(z))]) )]
              - E[log(1 - sigmoid(D(G(z)) - E[D(x)]))]
        """
        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify the generated batch images
        DG_score = self.model.D(G_output)

        # Compute D loss
        D_loss = -torch.mean(torch.log(torch.sigmoid(DX_score - DG_score.mean()) + 1e-8) \
                             + torch.log(torch.sigmoid(1 - DG_score) + 1e-8)) / 2

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: non-saturating loss for how well G(z) fools D,
            -E[log(sigmoid(D(G(z))-E[D(x)]))]
                -E[log(1-sigmoid(D(x)-E[D(G(z))]))]
        """
        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute the non-saturating loss for how D did versus the generations
        # of G
        G_loss = -torch.mean(torch.log(DG_score + 1e-8))

        return G_loss


if __name__ == '__main__':
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Initialize model
    model = RaNSGAN(Generator, Discriminator, image_size=(28, 28),
                    z_dim=20)

    # Initialize trainer
    trainer = RaNSGANTrainer(model=model,
                             train_iter=train_iter,
                             val_iter=val_iter,
                             test_iter=test_iter,
                             viz=False)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=2e-4,
                  D_lr=2e-4,
                  D_steps=1)
