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

from src.utils import *


class TrainerBase:

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input into the Generator G """
        return to_cuda(torch.randn(batch_size, z_dim))

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        images, _ = next(iter(iterator))
        images = to_cuda(images)
        return images

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)

    def generate_images(self, epoch, num_outputs=36, save=True):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = self.compute_noise(num_outputs, self.model.z_dim)

        # Transform noise to image
        images = self.model.G(noise)

        # Reshape to proper image size
        images = images.view(images.shape[0],
                             *self.model.image_shape
                             ).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1
        plt.show()

        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         % (epoch), nrow=grid_size)

    def viz_loss(self):
        """ Visualize loss for the generator, discriminator """

        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Dlosses,
                 'r')

        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title(self.name)
        plt.show()


