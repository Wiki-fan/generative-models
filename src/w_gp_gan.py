""" (WGPGAN) https://arxiv.org/abs/1701.07875
Wasserstein GAN with Gradient Penalties ('Improved Training of Wasserstein GANs')

The output of WGPGAN's D is unbounded unless passed through an activation
function. In this implementation, we use a ReLU activation function
as this empirically improves visualizations for binary MNIST.

WGAN GP roposes a gradient penalty to add to the WGAN discriminator loss as an
alternative method for enforcing the Lipschitz constraint (previously done via
weight clipping). This penalty does not suffer from the biasing of the
discriminator toward simple funtions due to weight clipping. Additionally, the
reformulation of the discriminator by adding a gradient penaltyterm makes batch
normalization not necessary. This is notable because batch normalization
implicitly changes the discriminator's problem from mapping one-to-one to
many-to-many.
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


class WGPGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_size, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_size, z_dim)

        class WGPGANDiscriminator(Discriminator):
            """ Critic (not trained to classify). Input is an image (real or generated),
                output is the approximate Wasserstein Distance between z~P(G(z)) and real.
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = WGPGANDiscriminator(image_size, output_dim)


class WGPGANTrainer(TrainerBase):
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

    def train(self, num_epochs, G_lr=1e-4, D_lr=1e-4, D_steps=5):
        """ Train a WGAN GP

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

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

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

    def train_D(self, images, LAMBDA=10):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images)  # D(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = to_var(torch.rand(images.shape[0], 1).unsqueeze(-1).unsqueeze(-1).expand(images.size()))
        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon * images + (1 - epsilon) * G_output.view(images.shape)  # TODO: no view
        D_interpolation = self.model.D(G_interpolation)

        # Compute the gradients of D with respect to the noise generated input
        weight = to_cuda(torch.ones(D_interpolation.size()))

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute WGAN-GP loss for G (same loss as WGAN)
        G_loss = -1 * (torch.mean(DG_score))

        return G_loss


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = WGPGAN(Generator, Discriminator, image_size=(28, 28),
                   z_dim=20)

    # Init trainer
    trainer = WGPGANTrainer(model=model,
                            train_iter=train_iter,
                            val_iter=val_iter,
                            test_iter=test_iter,
                            viz=False)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
