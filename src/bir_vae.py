""" (BIR-VAE) https://arxiv.org/abs/1807.07306
Bounded Information Rate Variational Autoencoder

This VAE variant makes a slight change to the original formulation
in an effort to enforce mutual information between our inputs x and the
latent space z. The change is setting the variance of q(z|x) instead of
learning it, which allows us to control the information rate across the
channel (Eqn. 7). It also implicity maximizes mutual information between
x and z without direct computation subject to the constraint q(z)=N(0,I).
This happens when the Maximum Mean Discrepancy between q(z) and p(z) is
0, and causes the mutual information term to reduce to a constant because
the differential entropy between h_q(z)[z] and h_q(z|x)[z] are both fixed
(Eqn. 10/11). The output of the decode is the mean of the isotropic
Gaussian with variance 1, so the log likelihood reduced to the negative
mean square error (i.e. we use MSELoss instead of NLLLoss).
"""

from copy import deepcopy

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import AutoencoderBase
from src.utils import *


class Encoder(nn.Module):
    """ MLP encoder for VAE. Input is an image, outputs are the mean and std of
    the latent representation z pre-reparametrization
    """

    def __init__(self, image_shape, z_dim):
        super().__init__()

        self.__dict__.update(locals())

        hidden_dim = 400
        self.linear = nn.Linear(np.prod(image_shape), hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        activated = F.relu(self.linear(x))
        mu = self.mu(activated)
        return mu


class Decoder(nn.Module):
    """ MLP decoder for VAE. Input is a reparametrized latent representation,
    output is reconstructed image
    """

    def __init__(self, z_dim, image_shape):
        super().__init__()

        self.__dict__.update(locals())

        hidden_dim = 400
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.recon = nn.Linear(hidden_dim, np.prod(image_shape))

    def forward(self, z):
        activated = F.relu(self.linear(z))
        reconstructed = torch.sigmoid(self.recon(activated))
        reconstructed = reconstructed.view((reconstructed.shape[0],) + self.image_shape)
        return reconstructed


class BIRVAE(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method. Parameter I indicates how many 'bits' should be let through.
    """

    def __init__(self, Encoder, Decoder, image_shape, z_dim=20, I=13.3):
        super().__init__()

        self.__dict__.update(locals())

        self.encoder = Encoder(image_shape=image_shape, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, image_shape=image_shape)

        self.set_var = 1 / (4**(I / z_dim))

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        out_img = self.decoder(z)
        return out_img, z

    def reparameterize(self, mu):
        """" Reparametrization trick: z = mean + epsilon, where epsilon ~ N(0, set_var)."""
        eps = to_cuda(torch.from_numpy(np.random.normal(loc=0.0,
                                                        scale=self.set_var,
                                                        size=mu.shape)).float())
        z = mu + eps  # Algorithm 1
        return z


class BIRVAETrainer(AutoencoderBase):
    def __init__(self, model, train_iter, val_iter, test_iter):
        """ Object to hold data iterators, train the model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.best_val_loss = 1e10
        self.debugging_image, _ = next(iter(test_iter))

        self.mmd_loss = []
        self.recon_loss = []
        self.num_epochs = 0

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5,
              writer=None, plot_to_screen=True, silent=False, with_progressbar=False, sample_interval=1):
        """ Train a Variational Autoencoder

            Logs progress using total loss, reconstruction loss, maximum mean
            discrepancy (MMD), and validation loss

        Inputs:
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer
            weight_decay: float, weight decay for Adam optimizer
        """
        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                       if p.requires_grad],
                               lr=lr,
                               weight_decay=weight_decay)

        # Begin training
        for epoch in tqdm(range(1, num_epochs + 1), disable=not with_progressbar):

            self.model.train()
            epoch_loss, epoch_recon, epoch_mmd = [], [], []

            for batch in self.train_iter:
                # Zero out gradients
                optimizer.zero_grad()

                # Compute mean squared error loss, mean maximum discrepancy loss
                mse_loss, mmd_loss = self.compute_batch(batch)
                batch_loss = mse_loss + mmd_loss

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log metrics
                epoch_loss.append(batch_loss.item())
                epoch_recon.append(mse_loss.item())
                epoch_mmd.append(mmd_loss.item())

            # Save progress
            self.mmd_loss.extend(epoch_mmd)
            self.recon_loss.extend(epoch_recon)

            # Test the model on the validation set
            self.model.eval()
            val_loss = self.evaluate(self.val_iter)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_model = deepcopy(self.model)
                self.best_val_loss = val_loss

            if not silent:
                # Progress logging
                print("Epoch[%d/%d], Total Loss: %.4f, MSE Loss: %.4f, MMD Loss: %.4f, Val Loss: %.4f"
                      % (epoch, num_epochs, np.mean(epoch_loss), np.mean(epoch_recon), np.mean(epoch_mmd), val_loss))

            if writer is not None:
                writer.add_scalar('Total Loss', np.mean(epoch_loss), epoch)
                writer.add_scalar('MSE Loss', np.mean(epoch_recon), epoch)
                writer.add_scalar('MMD Loss', np.mean(epoch_mmd), epoch)
                writer.add_scalar('Val Loss', val_loss, epoch)

            self.num_epochs += 1

            if epoch % sample_interval == 0:
                # Visualize autoencoder progress
                self.reconstruct_images(self.debugging_image, epoch, writer=writer, show=plot_to_screen)
                # Visualize generator progress
                self.sample_images(epoch, writer=writer, show=plot_to_screen)

    def compute_batch(self, batch, LAMBDA=1000.):
        """ Compute loss for a batch of examples

        LAMBDA: (float) a weighting factor for MMD loss vs. MSE loss
        """

        # Reshape images
        images, _ = batch
        images = to_cuda(images)

        # Get output images, mean, std of encoded space
        outputs, z = self.model(images)

        # Mean squared error loss
        mse_loss = torch.sum((images - outputs)**2)

        # Maximum mean discrepancy
        mmd_loss = LAMBDA * self.maximum_mean_discrepancy(z)

        return mse_loss, mmd_loss

    def maximum_mean_discrepancy(self, z):
        """ Maximum mean discrepancy of a Gaussian kernel """
        x = torch.randn(z.shape)
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(z, z)
        xy_kernel = self.compute_kernel(x, z)
        mmd_loss = x_kernel.sum() + y_kernel.sum() - 2 * xy_kernel.sum()
        return mmd_loss

    def compute_kernel(self, x, y):
        """ Compute Gaussian kernel for MMD (Eqn. 13) """
        # Get sizes, dimensions
        x_size, y_size, dim = x.size(0), y.size(0), x.size(1)

        # Unsqueeze and expand so we can compute element-wise operations
        x, y = x.unsqueeze(1), y.unsqueeze(0)
        tiled_x, tiled_y = x.expand(x_size, y_size, dim), y.expand(x_size, y_size, dim)

        # Compute Gaussian Kernel (Eqn. 13)
        kernel_input = torch.div(torch.mean(torch.pow(tiled_x - tiled_y, 2), dim=2), dim)
        return torch.exp(-kernel_input)

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        loss = []
        for batch in iterator:
            mse_loss, mmd_loss = self.compute_batch(batch)
            batch_loss = mse_loss + mmd_loss
            loss.append(batch_loss.item())

        loss = np.mean(loss)
        return loss

    def reconstruct(self, images):
        return self.model(to_cuda(images))[0]

    def explore_latent_space(self, BIRVAE, num_epochs=3):
        """ Viz method 3: train a VAE with 2 latent variables,
        compare variational means
        """

        # Initialize and train a VAE with size two dimension latent space
        train_iter, val_iter, test_iter = get_data()
        latent_model = BIRVAE(Encoder, Decoder, image_shape=self.model.image_shape, z_dim=2, I=13.3)
        latent_space = BIRVAETrainer(latent_model, train_iter, val_iter, test_iter)
        latent_space.train(num_epochs)
        latent_model = latent_space.best_model

        # Across batches in train iter, collect variational means
        data = []
        for batch in train_iter:
            images, labels = batch
            images = to_cuda(images)
            mu = latent_model.encoder(images)

            for label, (m1, m2) in zip(labels, mu):
                data.append((label.item(), m1.item(), m2.item()))

        self.plot_latent_space(latent_model, data)

        return latent_model

    def make_all(self):
        """ Execute all viz methods outlined in this class """

        print('Sampling images from latent space...')
        self.sample_images(save=False)

        print('Interpolating between two randomly sampled...')
        self.sample_interpolated_images()

        print('Exploring latent representations...')
        _ = self.explore_latent_space()

    def viz_loss(self):
        """ Visualize reconstruction loss """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot reconstruction loss in red
        plt.plot(np.linspace(1, self.num_epochs, len(self.recon_loss)),
                 self.recon_loss,
                 'r')

        # Plot KL-divergence in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.mmd_loss)),
                 self.mmd_loss,
                 'g')

        # Add legend, title
        plt.legend(['Reconstruction', 'Maximum Mean Discrepancy'])
        plt.title(self.name)
        plt.show()


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binzarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = BIRVAE(Encoder, Decoder,
                   image_shape=(1, 28, 28),
                   z_dim=20,
                   I=13.3)

    # Init trainer
    trainer = BIRVAETrainer(model=model,
                            train_iter=train_iter,
                            val_iter=val_iter,
                            test_iter=test_iter)

    # Train
    trainer.train(num_epochs=10,
                  lr=1e-3,
                  weight_decay=1e-5)

    trainer.explore_latent_space(BIRVAE)
    trainer.sample_interpolated_images()
