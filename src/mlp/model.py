import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.functional import F

from vae import VanillaVAE


class InterpMLP(pl.LightningModule):
    def __init__(
        self,
        vae_model: VanillaVAE,
        hidden_channels: list[int] = [100, 100, 100],
        **kwargs,
    ) -> None:
        super(InterpMLP, self).__init__()
        self.save_hyperparameters(ignore="vae_model")

        # Freeze VAE model when training MLP
        self.vae_model = vae_model
        self.vae_model.freeze()

        self.model_params = kwargs["model_params"]
        self.data_params = kwargs["data_params"]
        self.train_params = kwargs["train_params"]
        self.meta_dims = self.model_params["meta_dims"]

        in_channels = self.model_params["in_channels"]
        neighbor_size = self.data_params["neighbor_size"]
        out_channels = self.model_params["out_channels"]

        # MLP Model
        modules = []
        cur_features = in_channels * neighbor_size
        for h_dim in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Linear(cur_features, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            cur_features = h_dim

        self.first_layer = nn.Flatten()
        self.mlp_model = nn.Sequential(*modules, nn.Linear(cur_features, out_channels))

    def forward(self, input: Tensor, label: dict) -> list[Tensor]:
        mu, log_var = self.vae_model.encode(input[:, :, 3:].float())
        encoder_output = self.vae_model.reparameterize(mu, log_var)

        mlp_input = torch.cat([encoder_output, input[:, :, :3].float()], dim=2)
        flatten = self.first_layer(mlp_input)
        mlp_output = self.mlp_model(flatten)

        meta_data = [label[dim].unsqueeze(1).float() for dim in self.meta_dims]
        decoder_input = torch.concat([mlp_output, *meta_data], dim=1)
        decoder_output = self.vae_model.decode(decoder_input)

        return [decoder_output, label]

    def loss_function(self, *args, **kwargs) -> dict:
        output, label = args

        real = label["real"]
        real_filled = torch.where(torch.isnan(real), output, real)

        mse = F.mse_loss(output, real_filled)
        mae = F.l1_loss(output, real_filled)

        return {"loss": mse, "mae": mae}

    def training_step(self, batch):
        input, label = batch

        results = self.forward(input, label=label)
        train_loss = self.loss_function(*results)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss["loss"]

    def validation_step(self, batch):
        input, label = batch

        results = self.forward(input, label=label)
        val_loss = self.loss_function(*results)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.train_params["LR"],
            weight_decay=self.train_params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.train_params["scheduler_gamma"])

        return [optimizer], [scheduler]
