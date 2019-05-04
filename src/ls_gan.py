""" (LS GAN) https://arxiv.org/abs/1611.04076
Least Squares GAN

The output of LSGAN's D is unbounded unless passed through an activation
function. In this implementation, we include a sigmoid activation function as
this empirically improves visualizations for binary MNIST.

Tackles the vanishing gradients problem associated with GANs by swapping out
the cross entropy loss function with the least squares (L2) loss function.
The authors show that minimizing this objective is equivalent to minimizing the
Pearson chi-squared divergence. They claim that using the L2 loss function
penalizes samples that appear to be real to the discriminator, but lie far away
from the decision boundary. In this way, the generated images are made to appear
closer to real data. It also stabilizes training.
"""

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import TrainerBase
from src.utils import *


class LSGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, Generator, Discriminator, image_shape, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_shape, z_dim)

        class LSGANDiscriminator(Discriminator):
            """ Critic (not trained to classify). Input is an image (real or generated),
                output is approximate least-squares divergence.
            """

            def __init__(self, image_shape, output_dim):
                super().__init__(image_shape, output_dim)

            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        self.D = LSGANDiscriminator(image_shape, output_dim)


class LSGANTrainer(TrainerBase):
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

    def train(self, num_epochs, G_lr=1e-4, D_lr=1e-4, D_steps=1,
              writer=None, plot_to_screen=False, silent=True, sample_interval=1):
        """ Train a least-squares GAN with Gradient Penalty

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's Adam optimize
            D_lr: float, learning rate for discriminator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
        """
        # Initialize optimizers
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                         if p.requires_grad], lr=G_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                         if p.requires_grad], lr=D_lr)

        # Approximate steps/epoch given D_steps per epoch -->
        #  roughly train in the same way as if D_step (1) == G_step (1)
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

                    # Train D to approximate the distance between real, generated
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

                # Train the generator to (roughly) minimize the approximated
                # least-squares distance
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

    def train_D(self, images, a=0, b=1):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: L2 loss for discriminator,
            0.50 * E[(D(x) - a)^2] + 0.50 * E[(D(G(z)) - b)^2],
            where a and b are labels for generated (0) and real (1) data
        """
        # Sample noise, an output from the generator
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images)  # D(x)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute L2 loss for D
        D_loss = (0.50 * torch.mean((DX_score - b)**2)) \
                 + (0.50 * torch.mean((DG_score - a)**2))

        return D_loss

    def train_G(self, images, c=1):
        """ Run 1 step of training for generator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            G_loss: L2 loss for G,
            0.50 * E[(D(G(z)) - c)^2],
            where c is the label that G wants D to believe for fake data (1)
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute L2 loss for G
        G_loss = 0.50 * torch.mean((DG_score - c)**2)

        return G_loss


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = LSGAN(Generator, Discriminator, image_shape=(28, 28),
                  z_dim=20)

    # Init trainer
    trainer = LSGANTrainer(model=model,
                           train_iter=train_iter,
                           val_iter=val_iter,
                           test_iter=test_iter)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
