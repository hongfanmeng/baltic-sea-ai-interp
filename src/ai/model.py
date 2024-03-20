from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VanillaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        meta_dims: List = [],
        **kwargs,
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.meta_dims = meta_dims

        if hidden_dims is None:
            hidden_dims = [40, 40, 40]

        # Build Encoder
        modules = []
        cur_features = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur_features, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            cur_features = h_dim
        self.encoder = nn.Sequential(*modules)

        # mu, sigma
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim + len(meta_dims), hidden_dims[-1])
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], in_channels),
            # nn.BatchNorm1d(in_channels),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, meta: dict) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if self.meta_dims:
            z = torch.concat([z, *[meta[dim].unsqueeze(1).float() for dim in self.meta_dims]], dim=1)
        recons = self.decode(z)
        return [recons, input, mu, log_var, meta]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons, input, mu, log_var, meta = args[:5]

        real = meta["real"]
        real_filled = torch.where(torch.isnan(real), recons, real)

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, real_filled)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }
