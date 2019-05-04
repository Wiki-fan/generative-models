""" (FisherGAN) https://arxiv.org/abs/1606.07536
Fisher GAN

From the abstract:
"In this paper we introduce Fisher GAN which fits within the
Integral Probability Metrics (IPM) framework for training GANs.
Fisher GAN defines a critic with a data dependent constraint on
its second order moments. We show in this paper that Fisher GAN
allows for stable and time efficient training that does not
compromise the capacity of the critic, and does not need data
independent constraints such as weight clipping."

Integral Probability Metrics (IPM) framework simply means that
the outputs of the discriminator can be interpretted
probabilistically. This is similar to WGAN/WGAN-GP. Whereas
WGAN-GP uses a penalty on the gradients of the critic, FisherGAN
imposes a constraint on the second order moments of the critic.
Also, the Fisher IPM corresponds to the Chi-squared distance
between distributions.

The main empirical claims are that FisherGAN yields better
inception scores and has less computational overhead than WGAN.
"""

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class FisherGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class FisherGANDiscriminator(Discriminator):
            """ Discriminator. Input is an image (real or generated), output is P(generated).
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = FisherGANDiscriminator(image_shape, output_dim)


class FisherGANTrainer(TrainerBase):
    """ Object to hold data iterators, train a GAN variant
    """

    def __init__(self, model, train_iter, val_iter, test_iter):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.num_epochs = 0

    def train(self, num_epochs, G_lr=1e-4, D_lr=1e-4, D_steps=1, RHO=1e-6,
              writer=None, plot_to_screen=False, silent=True, sample_interval=1):
        """ Train FisherGAN using IPM framework

            Logs progress using G loss, D loss, G(x), D(G(x)),
            IPM ratio (want close to 0.50), Lambda (want close to 0),
            and visualizations of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
            LAMBDA: float, initial weight on constraint term
            RHO: float, quadratic penalty weight
        """
        # Initialize alpha
        self.LAMBDA = to_var(torch.zeros(1))
        self.RHO = to_var(torch.tensor(RHO))

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
                    D_loss, IPM_ratio = self.train_D(images)

                    # Update parameters
                    D_loss.backward()

                    # Minimize lambda for 'artisinal SGD'
                    self.LAMBDA = self.LAMBDA + self.RHO * self.LAMBDA.grad
                    self.LAMBDA = to_var(self.LAMBDA.detach())

                    # Now step optimizer
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                # So that G_loss and D_loss have the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the Generator to fool the discriminator
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
                print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f, IPM ratio: %.4f, Lambda: %.4f"
                      % (epoch, num_epochs, np.mean(G_losses), np.mean(D_losses),
                         IPM_ratio, self.LAMBDA))

            if writer is not None:
                writer.add_scalar('G_loss', np.mean(G_losses), epoch)
                writer.add_scalar('D_loss', np.mean(D_losses), epoch)
                writer.add_scalar('IPM_ratio', IPM_ratio, epoch)
                writer.add_scalar('Lambda', self.LAMBDA, epoch)

            self.num_epochs += 1

            if epoch % sample_interval == 0:
                # Visualize generator progress
                self.generate_images(epoch, writer=writer, show=plot_to_screen)

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: FisherGAN IPM loss (Equation 9 of paper)
        """
        # Generate labels (ones indicate real images, zeros indicate generated)
        X_labels = to_cuda(torch.ones(images.shape[0], 1))
        G_labels = to_cuda(torch.zeros(images.shape[0], 1))

        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z), discriminate D(G(z))
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)
        DG_score = self.model.D(G_output)

        # First and second order central moments (Gaussian assumed)
        DX_moment_1, DG_moment_1 = DX_score.mean(), DG_score.mean()
        DX_moment_2, DG_moment_2 = (DX_score**2).mean(), (DG_score**2).mean()

        # Compute constraint on second order moments
        OMEGA = 1 - (0.5 * DX_moment_2 + 0.5 * DG_moment_2)

        # Compute loss (Eqn. 9)
        D_loss = -((DX_moment_1 - DG_moment_1) \
                   + self.LAMBDA * OMEGA \
                   - (self.RHO / 2) * (OMEGA**2))

        # For progress logging
        IPM_ratio = DX_moment_1.item() - DG_moment_1.item() \
                    / 0.5 * (DX_moment_2.item() - DG_moment_2.item())**0.5

        return D_loss, IPM_ratio

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: FisherGAN IPM loss (Equation 9 of paper)
        """
        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute loss by minimizing mean difference
        G_loss = -DG_score.mean()

        return G_loss


if __name__ == '__main__':
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = FisherGAN(Generator, Discriminator, image_shape=(28, 28),
                      z_dim=20)
    # Init trainer
    trainer = FisherGANTrainer(model=model,
                               train_iter=train_iter,
                               val_iter=val_iter,
                               test_iter=test_iter)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
