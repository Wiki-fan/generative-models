import os
from itertools import product

import torchvision
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from src.utils import *


class Base:
    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input into the Generator G """
        return to_cuda(torch.randn(batch_size, z_dim))

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


class TrainerBase(Base):
    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        images, _ = next(iter(iterator))
        images = to_cuda(images)
        return images

    def generate_images(self, epoch, num_outputs=36, save=True, show=True, writer=None):
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

        if writer is not None:
            writer.add_image('generated', fig2rgba(), dataformats='HWC', global_step=epoch)

        if show:
            plt.show()
        else:
            plt.close()

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


class AutoencoderBase(Base):

    def reconstruct_images(self, images, epoch, save=True, show=True, writer=None):
        """ Sample images from latent space at each epoch """

        # Reshape images, pass through model, reshape reconstructed output
        batch = to_cuda(images)
        reconst_images = self.reconstruct(batch)
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

        image_grid = make_grid(reconst_images.data.view(*images.shape))

        if writer is not None:
            writer.add_image('reconstructed', fig2rgba(), dataformats='HWC', global_step=epoch)
            # writer.add_image('reconstructed', image_grid, global_step=epoch)

        if show:
            plt.show()
        else:
            plt.close()

        # Save
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.data,
                                         outname + 'real.png',
                                         nrow=grid_size)
            torchvision.utils.save_image(reconst_images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png' % (epoch),
                                         nrow=grid_size)

    def sample_images(self, epoch, num_images=36, save=True, writer=None, show=False):
        """ Viz method 1: generate images by sampling z ~ p(z), x ~ p(x|z,θ) """

        # Sample z
        z = self.compute_noise(num_images, self.model.z_dim)

        # Pass into decoder
        sample = self.model.decoder(z)

        # Plot
        image_grid = make_grid(sample.data.view(num_images,
                                                *self.model.image_shape),
                               nrow=int(num_images**0.5))
        img = ToPILImage()(image_grid)
        plt.imshow(img)

        if writer is not None:
            # writer.add_image('generated', fig2rgba(), global_step=epoch)
            writer.add_image('generated', image_grid, global_step=epoch)

        if show:
            plt.show()
        else:
            plt.close()

        # Save
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            img.save(outname + 'sample_%d.png' % (epoch))

    def sample_interpolated_images(self, n=9):
        """ Viz method 2: sample two random latent vectors from p(z),
        then sample from their interpolated values
        """

        # Sample latent vectors
        z1 = self.compute_noise(1, self.model.z_dim)
        z2 = self.compute_noise(1, self.model.z_dim)

        # Interpolate within latent vectors
        samples = []
        for alpha in np.linspace(0, 1, n):
            z = to_cuda(alpha * z1 + (1 - alpha) * z2)
            sample = self.model.decoder(z)[0]
            samples.append(sample)

        samples_tensor = torch.stack(samples)

        image_grid = make_grid(samples_tensor.data.view(n, *self.model.image_shape), nrow=int(n**0.5))
        plt.imshow(ToPILImage()(image_grid))

        plt.show()

    def plot_latent_space(self, latent_model, data):
        # Plot
        labels, m1s, m2s = zip(*data)
        plt.figure(figsize=(10, 10))
        plt.scatter(m1s, m2s, c=labels)
        plt.legend([str(i) for i in set(labels)])
        plt.show()

        # Evenly sample across latent space, visualize the outputs
        mu = torch.stack([torch.FloatTensor([m1, m2])
                          for m1 in np.linspace(-2, 2, 10)
                          for m2 in np.linspace(-2, 2, 10)])
        samples = latent_model.decoder(to_cuda(mu))
        image_grid = make_grid(samples.data.view(mu.shape[0],
                                                 *self.model.image_shape
                                                 ), nrow=10)
        plt.imshow(ToPILImage()(image_grid))

        plt.show()
