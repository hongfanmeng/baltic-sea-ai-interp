from typing import List

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F


class VanillaVAE(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.save_hyperparameters()
        self.train_params: dict = kwargs["train_params"]
        self.model_params: dict = kwargs["model_params"]

        self.latent_dim: int = self.model_params["latent_dim"]
        self.meta_dims: list[str] = self.model_params["meta_dims"]

        in_channels: int = self.model_params["in_channels"]
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
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim + len(self.meta_dims), hidden_dims[-1])
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
        kld_weight = kwargs["M_N"]

        # loss only calc on non-nan values
        mask = ~torch.isnan(real)
        masked_recons = recons[mask]
        masked_real = real[mask]

        recons_loss = F.mse_loss(masked_recons, masked_real)
        recons_mae = F.l1_loss(masked_recons, masked_real)
        recons_mape = torch.mean(torch.abs((masked_recons - masked_real) / masked_real))

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "recons_mse": recons_loss.detach(),
            "recons_mae": recons_mae.detach(),
            "recons_mape": recons_mape.detach(),
            "kld": -kld_loss.detach(),
        }

    def training_step(self, batch, batch_idx):
        input, meta = batch

        results = self.forward(input, meta=meta)
        train_loss = self.loss_function(*results, M_N=self.train_params["kld_weight"])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx):
        input, meta = batch

        results = self.forward(input, meta=meta)
        val_loss = self.loss_function(*results, M_N=1.0)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input, meta = batch

        results = self.forward(input, meta=meta)
        test_loss = self.loss_function(*results, M_N=1.0)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.train_params["LR"],
            weight_decay=self.train_params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.train_params["scheduler_gamma"])

        return [optimizer], [scheduler]
