""" (MM GAN) https://arxiv.org/abs/1406.2661
Mini-max GAN

From the abstract: 'We propose a new framework for estimating generative models
via an adversarial process, in which we simultaneously train two models: a
generative model G that captures the data distribution, and a discriminative
model D that estimates the probability that a sample came from the training data
rather than G. The training procedure for G is to maximize the probability of D
making a mistake.'

Compared to MM GAN, the only change is the generator's loss.

NS GAN: L(G) = -E[log(D(G(z)))]
MM GAN: L(G) =  E[log(1-D(G(z)))]

In both NS GAN and MM GAN, the output of G can be interpretted as a probability.
"""

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class MMGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class MMGANDiscriminator(Discriminator):
            """ Discriminator. Input is an image (real or generated), output is P(generated).
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = MMGANDiscriminator(image_shape, output_dim)


class MMGANTrainer(TrainerBase):
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

    def train(self, num_epochs, G_lr=2e-4, D_lr=2e-4, D_steps=1, G_init=5,
              writer=None, plot_to_screen=True, silent=False, with_progressbar=False, sample_interval=1):
        """ Train a vanilla GAN using minimax gradients loss for the generator.

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
            G_init: int, number of training steps to pre-train G for
        """
        # Initialize optimizers
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                         if p.requires_grad], lr=G_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters() if p.requires_grad], lr=D_lr)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))

        # Let G train for a few steps before beginning to jointly train G
        # and D because MM GANs have trouble learning very early on in training
        if G_init > 0:
            for _ in range(G_init):
                # Process a batch of images
                images = self.process_batch(self.train_iter)

                # Zero out gradients for G
                G_optimizer.zero_grad()

                # Pre-train G
                G_loss = self.train_G(images)

                # Backpropagate the generator network
                G_loss.backward()
                G_optimizer.step()

            print('G pre-trained for {0} training steps.'.format(G_init))
        else:
            print('G not pre-trained -- GAN unlikely to converge.')

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

                        # Train D to learn to discriminate between real and generated images
                        D_loss = self.train_D(images)

                        # Update parameters
                        D_loss.backward()
                        D_optimizer.step()

                        # Log results, backpropagate the discriminator network
                        D_step_loss.append(D_loss.item())

                    # We report D_loss in this way so that G_loss and D_loss have the same number of entries.
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
                    self.model.generate_images(epoch, writer=writer, show=plot_to_screen)

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: non-saturing loss for discriminator,
            -E[log(D(x))] - E[log(1 - D(G(z)))]
        """
        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify the fake batch images, get the loss for these using sigmoid cross entropy
        DG_score = self.model.D(G_output)

        # Compute vanilla (original paper) D loss
        D_loss = torch.sum(-torch.mean(torch.log(DX_score + 1e-8)
                                       + torch.log(1 - DG_score + 1e-8)))

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: minimax loss for how well G(z) fools D,
            -E[log(D(G(z)))]
        """
        # Get noise (denoted z), classify it using G, then classify the output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute the minimax loss for how D did versus the generations of G using sigmoid cross entropy
        G_loss = torch.mean(torch.log((1 - DG_score) + 1e-8))

        return G_loss


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = MMGAN(Generator, Discriminator, image_shape=(28, 28),
                  z_dim=20)

    # Init trainer
    trainer = MMGANTrainer(model=model,
                           train_iter=train_iter,
                           val_iter=val_iter,
                           test_iter=test_iter,
                           viz=False)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=2e-4,
                  D_lr=2e-4,
                  D_steps=1,
                  G_init=5)
