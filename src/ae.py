""" (AE)
Standard Autoencoder

Autoencoders take an input representation, encode it into a reduced
dimensionality space using an 'encoder network.' They then decode this
encoded representation by using a 'decoder network,' which transforms it
back to its original representation.
"""

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from tqdm import tqdm
from itertools import product

from src.utils import *


class Encoder(nn.Module):
    """ Feedforward network encoder. Input is an image, output is encoded
    vector representation of that image.
    """

    def __init__(self, image_shape, hidden_dim):
        super().__init__()

        self.__dict__.update(locals())

        self.linear = nn.Linear(np.prod(image_shape), hidden_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return F.relu(self.linear(x))


class Decoder(nn.Module):
    """ Feedforward network decoder. Input is an encoded vector representation,
    output is reconstructed image.
    """

    def __init__(self, hidden_dim, image_shape):
        super().__init__()

        self.__dict__.update(locals())

        self.linear = nn.Linear(hidden_dim, np.prod(image_shape))

    def forward(self, encoder_output):
        decoded = torch.sigmoid(self.linear(encoder_output))
        return decoded.view((decoded.shape[0],) + self.image_shape)


class Autoencoder(nn.Module):
    """ Autoencoder super class to encode then decode an image
    """

    def __init__(self, image_shape=(28,28), hidden_dim=32):
        super().__init__()

        self.__dict__.update(locals())

        self.encoder = Encoder(image_shape=image_shape, hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, image_shape=image_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderTrainer:
    def __init__(self, model, train_iter, val_iter, test_iter):
        """ Object to hold data iterators, train the model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.best_val_loss = 1e10
        self.debugging_image, _ = next(iter(test_iter))

        self.recon_loss = []
        self.num_epochs = 0

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5,
              writer=None, plot_to_screen=True, silent=False, with_progressbar=False, sample_interval=1):
        """ Train an Autoencoder
            Logs progress using total loss, reconstruction loss, kl_divergence,
            and validation loss

        Inputs:
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer (default 1e-3)
            weight_decay: float, weight decay for Adam optimizer (default 1e-5)
        """

        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                       if p.requires_grad],
                               lr=lr,
                               weight_decay=weight_decay)

        # Begin training
        for epoch in tqdm(range(1, num_epochs + 1)):

            self.model.train()
            epoch_loss = []

            for batch in self.train_iter:
                # Zero out gradients
                optimizer.zero_grad()

                # Compute reconstruction loss (binary cross entropy)
                batch_loss = self.compute_batch(batch)

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log progress metrics
                epoch_loss.append(batch_loss.item())

            # Save progress
            self.recon_loss.extend(epoch_loss)

            # Evaluate the model on the validation set
            self.model.eval()
            val_loss = self.evaluate(self.val_iter)

            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_model = deepcopy(self.model)
                self.best_val_loss = val_loss

            if not silent:
                # Print progress
                print("Epoch[%d/%d], Train Loss: %.4f, Val Loss: %.4f"
                      % (epoch, num_epochs, np.mean(epoch_loss), val_loss))

            if writer is not None:
                writer.add_scalar('Train_loss', np.mean(epoch_loss), epoch)
                writer.add_scalar('Val_loss', val_loss, epoch)

            self.num_epochs += 1

            # Debugging and visualization purposes
            if epoch % sample_interval == 0:
                self.reconstruct_images(self.debugging_image, epoch, writer=writer, show=plot_to_screen)

    def compute_batch(self, batch):
        """ Compute loss for a batch of examples """

        # Process a batch
        images, _ = batch
        images = to_cuda(images)

        # Encode the image and then decode the representation
        outputs = self.model(images)

        # L2 (mean squared error) loss.
        recon_loss = torch.sum((images - outputs)**2)

        return recon_loss

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        return np.mean([self.compute_batch(batch).item() for batch in iterator])

    def reconstruct_images(self, images, epoch, save=True, writer=None, show=False):
        """ Reconstruct a fixed input at each epoch for progress viz """
        # Pass through model, reshape reconstructed output
        batch = to_cuda(images)
        reconst_images = self.model(batch)
        reconst_images = reconst_images.view(images.shape).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(reconst_images.shape[0]**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].imshow(reconst_images[k].data.numpy(), cmap='gray')
            k += 1

        if writer is not None:
            writer.add_image('reconstructed', fig2rgba(), dataformats='HWC', global_step=epoch)
            writer.add_image('real', self.debugging_image.data, dataformats='HWC', global_step=epoch)

        if show:
            plt.show()
        else:
            plt.close()

        # Save
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(self.debugging_image.data,
                                         outname + 'real.png',
                                         nrow=grid_size)
            torchvision.utils.save_image(reconst_images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png' % (epoch),
                                         nrow=grid_size)

    def viz_loss(self):
        """ Visualize reconstruction loss """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot reconstruction loss in red
        plt.plot(np.linspace(1, self.num_epochs, len(self.recon_loss)),
                 self.recon_loss,
                 'r')

        # Add legend, title
        plt.legend(['Reconstruction loss'])
        plt.title(self.name)
        plt.show()



if __name__ == '__main__':
    from src.mnist_utils import *

    # Load in binzarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = Autoencoder(image_shape=(1, 28, 28),
                        hidden_dim=32)

    # Init trainer
    trainer = AutoencoderTrainer(model=model,
                                 train_iter=train_iter,
                                 val_iter=val_iter,
                                 test_iter=test_iter)

    # Train
    trainer.train(num_epochs=5,
                  lr=1e-3,
                  weight_decay=1e-5,
                  plot_to_screen=True,
                  silent=False)
