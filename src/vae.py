""" (VAE) https://arxiv.org/abs/1312.6114
Variational Autoencoder

From the abstract:

"We introduce a stochastic variational inference and learning algorithm that
scales to large datasets and, under some mild differentiability conditions,
even works in the intractable case. Our contributions is two-fold. First, we
show that a reparameterization of the variational lower bound yields a lower
bound estimator that can be straightforwardly optimized using standard
stochastic gradient methods. Second, we show that for i.i.d. datasets with
continuous latent variables per datapoint, posterior inference can be made
especially efficient by fitting an approximate inference model (also called a
recognition model) to the intractable posterior using the proposed lower bound
estimator."

Basically VAEs encode an input into a given dimension z, reparametrize that z
using it's mean and std, and then reconstruct the image from reparametrized z.
This lets us tractably model latent representations that we may not be
explicitly aware of that are in the data. For a simple example of what this may
look like, read up on "Karl Pearson's Crabs." The basic idea was that a
scientist collected data on a population of crabs, noticed that the distribution
was non-normal, and Pearson postulated it was because there were likely more
than one population of crabs studied. This would've been a latent variable,
since the data collector did not initially know or perhaps even suspect this.
"""

from copy import deepcopy

import torch.optim as optim
from tqdm import tqdm

from src.trainer_base import AutoencoderBase
from src.utils import *


class Encoder(nn.Module):
    """ MLP encoder for VAE. Input is an image,
    outputs are the mean, std of the latent variable z pre-reparametrization
    """

    def __init__(self, image_shape, z_dim):
        super().__init__()

        self.__dict__.update(locals())

        hidden_dim = 400
        self.linear = nn.Linear(np.prod(image_shape), hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        activated = F.relu(self.linear(x))
        mu, log_var = self.mu(activated), self.log_var(activated)
        return mu, log_var


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


class VAE(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method for latent variable z
    """

    def __init__(self, Encoder, Decoder, image_shape, z_dim):
        super().__init__()

        self.__dict__.update(locals())

        self.encoder = Encoder(image_shape=image_shape, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, image_shape=image_shape)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out_img = self.decoder(z)
        return out_img, mu, log_var

    def reparameterize(self, mu, log_var):
        """" Reparametrization trick: z = mean + std*epsilon,
        where epsilon ~ N(0, 1).
        """
        epsilon = to_cuda(torch.randn(mu.shape))
        z = mu + epsilon * torch.exp(log_var / 2)  # 2 for convert var to std
        return z


class VAETrainer(AutoencoderBase):
    def __init__(self, model, train_iter, val_iter, test_iter):
        """ Object to hold data iterators, train the model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.best_val_loss = 1e10
        self.debugging_image, _ = next(iter(test_iter))

        self.kl_loss = []
        self.recon_loss = []
        self.num_epochs = 0

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5,
              writer=None, plot_to_screen=True, silent=False, with_progressbar=False, sample_interval=1):
        """ Train a Variational Autoencoder

            Logs progress using total loss, reconstruction loss, kl_divergence,
            and validation loss

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
            epoch_loss, epoch_recon, epoch_kl = [], [], []

            for batch in self.train_iter:
                # Zero out gradients
                optimizer.zero_grad()

                # Compute reconstruction loss, Kullback-Leibler divergence
                # for a batch for the variational lower bound (ELBO)
                recon_loss, kl_diverge = self.compute_batch(batch)
                batch_loss = recon_loss + kl_diverge

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log metrics
                epoch_loss.append(batch_loss.item())
                epoch_recon.append(recon_loss.item())
                epoch_kl.append(kl_diverge.item())

            # Save progress
            self.kl_loss.extend(epoch_kl)
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
                print("Epoch[%d/%d], Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.7f, Val Loss: %.4f"
                      % (epoch, num_epochs, np.mean(epoch_loss),
                         np.mean(epoch_recon), np.mean(epoch_kl), val_loss))

            if writer is not None:
                writer.add_scalar('Total Loss', np.mean(epoch_loss), epoch)
                writer.add_scalar('Reconst Loss', np.mean(epoch_recon), epoch)
                writer.add_scalar('KL Divergence', np.mean(epoch_kl), epoch)
                writer.add_scalar('Val Loss', val_loss, epoch)

            self.num_epochs += 1

            # Debugging and visualization purposes
            if epoch % sample_interval == 0:
                # Visualize autoencoder progress
                self.reconstruct_images(self.debugging_image, epoch, writer=writer, show=plot_to_screen)
                # Visualize generator progress
                self.sample_images(epoch, writer=writer, show=plot_to_screen)

    def compute_batch(self, batch):
        """ Compute loss for a batch of examples """
        # Reshape images
        images, _ = batch
        images = to_cuda(images)

        # Get output images, mean, std of encoded space
        outputs, mu, log_var = self.model(images)

        # L2 (mean squared error) loss
        recon_loss = torch.sum((images - outputs)**2)

        # Kullback-Leibler divergence between encoded space, Gaussian
        kl_diverge = self.kl_divergence(mu, log_var)

        return recon_loss, kl_diverge

    def kl_divergence(self, mu, log_var):
        """ Compute Kullback-Leibler divergence """
        return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var - 1))

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        loss = []
        for batch in iterator:
            recon_loss, kl_diverge = self.compute_batch(batch)
            batch_loss = recon_loss + kl_diverge
            loss.append(batch_loss.item())

        loss = np.mean(loss)
        return loss

    def reconstruct(self, images):
        return self.model(to_cuda(images))[0]

    def explore_latent_space(self, VAE, num_epochs=3):
        """ Viz method 3: train a VAE with 2 latent variables,
        compare variational means
        """

        # Initialize and train a VAE with size two dimension latent space
        train_iter, val_iter, test_iter = get_data()
        latent_model = VAE(Encoder, Decoder, image_shape=self.model.image_shape, z_dim=2)
        latent_space = VAETrainer(latent_model, train_iter, val_iter, test_iter)
        latent_space.train(num_epochs)
        latent_model = latent_space.best_model

        # Across batches in train iter, collect variational means
        data = []
        for batch in train_iter:
            images, labels = batch
            images = to_cuda(images)
            mu, log_var = latent_model.encoder(images)

            for label, (m1, m2) in zip(labels, mu):
                data.append((label.item(), m1.item(), m2.item()))

        self.plot_latent_space(latent_model, data)

        return latent_model

    def make_all(self):
        """ Execute all latent space viz methods outlined in this class """

        print('Sampled images from latent space:')
        self.sample_images(save=False)

        print('Interpolating between two randomly sampled')
        self.sample_interpolated_images()

        print('Exploring latent representations')
        _ = self.explore_latent_space()

    def viz_loss(self):
        """ Visualize reconstruction loss """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot reconstruction loss in red, KL divergence in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.recon_loss)),
                 self.recon_loss,
                 'r')
        plt.plot(np.linspace(1, self.num_epochs, len(self.kl_loss)),
                 self.kl_loss,
                 'g')

        # Add legend, title
        plt.legend(['Reconstruction', 'Kullback-Leibler'])
        plt.title(self.name)
        plt.show()


if __name__ == "__main__":
    from src.mnist_utils import *

    # Load in binzarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = VAE(Encoder, Decoder,
                image_shape=(1, 28, 28),
                z_dim=20)

    # Init trainer
    trainer = VAETrainer(model=model,
                         train_iter=train_iter,
                         val_iter=val_iter,
                         test_iter=test_iter)

    # Train
    trainer.train(num_epochs=5,
                  lr=1e-3,
                  weight_decay=1e-5,
                  plot_to_screen=True,
                  silent=False)

    trainer.explore_latent_space(VAE)
    trainer.sample_interpolated_images()
