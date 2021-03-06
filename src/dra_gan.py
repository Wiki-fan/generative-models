""" (DRAGAN) https://arxiv.org/abs/1705.07215
Deep Regret Analytic GAN

The output of DRAGAN's D can be interpreted as a probability, similarly to
MMGAN and NSGAN. DRAGAN is similar to WGANGP, but seems less stable.

Proposes to study GANs from a regret minimization perspective. This
model is very similar to WGAN GP, in that it is applying a gradient penalty to
try and get at an improved training objective based on how D and G would
optimally perform. They apply the gradient penalty only close to the real data
manifold (whereas WGAN GP picks the gradient location on a random line between
a real and randomly generated fake sample). For further details, see
Section 2.5 of the paper.
"""

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class DRAGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class DRAGANDiscriminator(Discriminator):
            """ Critic (not trained to classify). Input is an image (real or generated),
                output is the approximate Wasserstein Distance between z~P(G(z)) and real.
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = DRAGANDiscriminator(image_shape, output_dim)


class DRAGANTrainer(TrainerBase):
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

    def train(self, num_epochs, G_lr=1e-4, D_lr=1e-4, D_steps=5,
              writer=None, plot_to_screen=True, silent=False, with_progressbar=False, sample_interval=1):
        """ Train a Deep Regret Analytic GAN

            Logs progress using G loss, D loss, G(x), D(G(x)),
            visualizations of Generator output.

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
        with tqdm(range(1, num_epochs + 1), disable=not with_progressbar) as t:
            for epoch in t:
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

                    # We report D_loss in this way so that G_loss and D_loss
                    # have the same number of entries.
                    D_losses.append(np.mean(D_step_loss))

                    # TRAINING G: Zero out gradients for G
                    G_optimizer.zero_grad()

                    # Train the generator
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

    def train_D(self, images, LAMBDA=10, K=1, C=1):
        """ Run 1 step of training for discriminator

        Input:
            model: model instantiation
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: DRAGAN loss for discriminator,
            -E[log(D(x))] - E[log(1 - D(G(z)))] + λE[(||∇ D(G(z))|| - 1)^2]
        """
        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify generated batch images
        DG_score = self.model.D(G_output)

        # Compute vanilla (original paper) D loss
        D_loss = -torch.mean(torch.log(DX_score + 1e-8) \
                             + torch.log(1 - DG_score + 1e-8))

        # GRADIENT PENALTY STEPS:
        # Uniformly sample along one straight line per each batch entry.
        delta = to_cuda(torch.rand(images.shape[0], 1).unsqueeze(-1).unsqueeze(-1).expand(images.size()))

        # Generate images from the noise, ensure unit
        G_interpolation = to_var(delta * images.data + (1 - delta) *
                                 (images.data + C * images.data.std() \
                                  * to_cuda(torch.rand(images.size()))))

        # Discriminate generator interpolation
        D_interpolation = self.model.D(G_interpolation)
        interp_shape = D_interpolation.shape

        # Compute the gradients of D with respect to the noise generated input
        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=to_cuda(torch.ones(interp_shape)),
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - K)**2)

        # Compute DRAGAN loss for D
        D_loss = D_loss + grad_penalty

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            G_loss: DRAGAN (non-saturating) loss for G, -E[log(D(G(z)))]
        """
        # Get noise (denoted z), classify it using G, then classify the
        # output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute the non-saturating loss for how D did versus the generations
        # of G using sigmoid cross entropy
        G_loss = -torch.mean(torch.log(DG_score + 1e-8))

        return G_loss


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = DRAGAN(Generator, Discriminator, image_shape=(28, 28),
                   z_dim=20)

    # Init trainer
    trainer = DRAGANTrainer(model=model,
                            train_iter=train_iter,
                            val_iter=val_iter,
                            test_iter=test_iter)

    # Train
    trainer.train(num_epochs=65,
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
