import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torch.functional import F

from vae import VanillaVAE


class InterpRNN(pl.LightningModule):
    def __init__(
        self,
        vae_model: VanillaVAE,
        hidden_channels: list[int] = [100, 100, 100],
        **kwargs,
    ) -> None:
        super(InterpRNN, self).__init__()
        self.save_hyperparameters(ignore="vae_model")

        # Freeze VAE model when training MLP
        self.vae_model = vae_model
        self.vae_model.freeze()

        self.model_params = kwargs["model_params"]
        self.data_params = kwargs["data_params"]
        self.train_params = kwargs["train_params"]
        self.meta_dims = self.model_params["meta_dims"]

        in_channels = self.model_params["in_channels"]
        out_channels = self.model_params["out_channels"]
        neighbor_size = self.data_params["neighbor_size"]
        past_years = self.data_params["past_years"]

        # RNN Model
        self.rnn_input_layer = nn.Flatten(start_dim=2)
        self.rnn_model = nn.LSTM(
            input_size=in_channels * neighbor_size,
            hidden_size=100,
            num_layers=3,
            batch_first=True,
        )
        self.rnn_output_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LeakyReLU(),
            nn.Linear(100 * past_years, out_channels),
        )

    def forward(self, input: Tensor, label: dict) -> list[Tensor]:
        """
        Args:
            input (Tensor): (batch_size, year, neighbor, features), features[:3] = (dx, dy, dz), features[3:] = oxy
            label (dict): keys: year, lat, lon, max_dep, real, x, y, z
        """
        mu, log_var = self.vae_model.encode(input[:, :, :, 3:].float())
        encoder_output = self.vae_model.reparameterize(mu, log_var)

        rnn_input = torch.cat([encoder_output, input[:, :, :, :3].float()], dim=3)
        rnn_input = self.rnn_input_layer(rnn_input)

        rnn_output, _ = self.rnn_model(rnn_input)
        rnn_output = self.rnn_output_layer(rnn_output)

        meta_data = [label[dim].unsqueeze(1).float() for dim in self.meta_dims]
        decoder_input = torch.concat([rnn_output, *meta_data], dim=1)
        decoder_output = self.vae_model.decode(decoder_input)

        return [decoder_output, label]

    def loss_function(self, *args, **kwargs) -> dict:
        output, label = args

        real = label["real"]

        # loss only calc on non-nan values
        mask = ~torch.isnan(real)
        masked_output = output[mask]
        masked_real = real[mask]

        mse = F.mse_loss(masked_output, masked_real)
        mae = F.l1_loss(masked_output, masked_real)
        mape = torch.mean(torch.abs((masked_output - masked_real) / masked_real))

        return {"loss": mse, "mae": mae, "mape": mape}

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