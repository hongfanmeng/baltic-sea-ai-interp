import pytorch_lightning as pl
from torch import optim

from .model import VanillaVAE


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: VanillaVAE, **kwargs) -> None:
        super(VAEXperiment, self).__init__()
        self.save_hyperparameters(ignore="vae_model")

        self.model = vae_model
        self.params = self.hparams.exp_params

    def training_step(self, batch, batch_idx):
        input, meta = batch

        results = self.model.forward(input, meta=meta)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
            batch_idx=batch_idx,
        )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx):
        input, meta = batch

        results = self.model.forward(input, meta=meta)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            batch_idx=batch_idx,
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params["scheduler_gamma"])

        return [optimizer], [scheduler]
