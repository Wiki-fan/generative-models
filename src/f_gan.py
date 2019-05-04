""" (f-GAN) https://arxiv.org/abs/1606.00709
f-Divergence GANs

The authors empirically demonstrate that when the generative model is
misspecified and does not contain the true distribution, the divergence
function used for estimation has a strong influence on which model is
learned. To address this issue, they theoretically show that the
generative-adversarial approach is a special case of an existing, more
general variational divergence estimation approach and that any
f-divergence can be used for training generative neural samplers (which
are defined as models that take a random input vector and produce a sample
from a probability distribution defined by the network weights). They
then empirically show the effect of using different training
divergences on a trained model's average log likelihood of sampled data.

They test (forward) Kullback-Leibler, reverse Kullback-Leibler, Pearson
chi-squared, Neyman chi-squared, squared Hellinger, Jensen-Shannon,
and Jeffrey divergences.

We exclude Neyman and Jeffrey due to poor performance and nontrivial
implementations to yield 'convergence' (see scipy.special.lambertw
for how to implement Jeffrey, and Table 6 of Appendix C of the paper
for how to implement Neyman)
"""

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class fGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class fGANDiscriminator(Discriminator):
            """ Discriminator. Input is an image (real or generated), output is P(generated).
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = fGANDiscriminator(image_shape, output_dim)


class Divergence:
    """ Compute G and D loss using an f-divergence metric.
    Implementations based on Table 6 (Appendix C) of the arxiv paper.
    """

    def __init__(self, method):
        self.method = method.lower().strip()
        assert self.method in ['total_variation',
                               'forward_kl',
                               'reverse_kl',
                               'pearson',
                               'hellinger',
                               'jensen_shannon'], \
            'Invalid divergence.'

    def D_loss(self, DX_score, DG_score):
        """ Compute batch loss for discriminator using f-divergence metric """

        if self.method == 'total_variation':
            return -(torch.mean(0.5 * torch.tanh(DX_score)) \
                     - torch.mean(0.5 * torch.tanh(DG_score)))

        elif self.method == 'forward_kl':
            return -(torch.mean(DX_score) - torch.mean(torch.exp(DG_score - 1)))

        elif self.method == 'reverse_kl':
            return -(torch.mean(-torch.exp(DX_score)) - torch.mean(-1 - DG_score))

        elif self.method == 'pearson':
            return -(torch.mean(DX_score) - torch.mean(0.25 * DG_score**2 + DG_score))

        elif self.method == 'hellinger':
            return -(torch.mean(1 - torch.exp(DX_score)) \
                     - torch.mean((1 - torch.exp(DG_score)) / (torch.exp(DG_score))))

        elif self.method == 'jensen_shannon':
            return -(torch.mean(torch.tensor(2.) - (1 + torch.exp(-DX_score))) \
                     - torch.mean(-(torch.tensor(2.) - torch.exp(DG_score))))

    def G_loss(self, DG_score):
        """ Compute batch loss for generator using f-divergence metric """

        if self.method == 'total_variation':
            return -torch.mean(0.5 * torch.tanh(DG_score))

        elif self.method == 'forward_kl':
            return -torch.mean(torch.exp(DG_score - 1))

        elif self.method == 'reverse_kl':
            return -torch.mean(-1 - DG_score)

        elif self.method == 'pearson':
            return -torch.mean(0.25 * DG_score**2 + DG_score)

        elif self.method == 'hellinger':
            return -torch.mean((1 - torch.exp(DG_score)) / (torch.exp(DG_score)))

        elif self.method == 'jensen_shannon':
            return -torch.mean(-(torch.tensor(2.) - torch.exp(DG_score)))


class fGANTrainer(TrainerBase):
    """ Object to hold data iterators, train a GAN variant
    """

    def __init__(self, model, train_iter, val_iter, test_iter):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__ + '/' + self.loss_fnc.method

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.num_epochs = 0

    def train(self, num_epochs, method, G_lr=1e-4, D_lr=1e-4, D_steps=1,
              writer=None, plot_to_screen=False, silent=True, sample_interval=1):
        """ Train a standard vanilla GAN architecture using f-divergence as loss

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            method: str, divergence metric to optimize
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminsator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
        """
        # Initialize loss, indicate which GAN it is
        self.loss_fnc = Divergence(method)

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

                    # Train D to discriminate between real and generated images
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

                # Train G to generate images that fool the discriminator
                G_loss = self.train_G(images)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)

            if not silent:
                # Progress logging
                print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                      % (epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))

            if writer is not None:
                writer.add_scalar('G_loss', np.mean(G_losses), epoch)
                writer.add_scalar('D_loss', np.mean(D_losses), epoch)

            self.num_epochs += 1

            if epoch % sample_interval == 0:
                # Visualize generator progress
                self.generate_images(epoch, writer=writer, show=plot_to_screen)

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: f-divergence between generated, true distributions
        """
        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify the fake batch images, get the loss for these using sigmoid cross entropy
        DG_score = self.model.D(G_output)

        # Compute f-divergence loss
        D_loss = self.loss_fnc.D_loss(DX_score, DG_score)

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: f-divergence for difference between generated, true distributiones
        """
        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute f-divergence loss
        G_loss = self.loss_fnc.G_loss(DG_score)

        return G_loss


if __name__ == '__main__':
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = fGAN(Generator, Discriminator, image_shape=(28, 28),
                 z_dim=20)

    # Init trainer
    trainer = fGANTrainer(model=model,
                          train_iter=train_iter,
                          val_iter=val_iter,
                          test_iter=test_iter)
    # Train
    trainer.train(num_epochs=25,
                  method='jensen_shannon',
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
