import os
import warnings
from pathlib import Path

import yaml
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rnn import InterpRNN, RNNDataModule
from vae import VanillaVAE


class UpdateParams(Callback):
    """
    Performance optimization for DataLoader

    epoch 0 using multiple workers to load and cache data,

    set num_workers to 0 after epoch 0 to load cached data on main process
    """

    def on_train_epoch_end(self, trainer: Trainer, _) -> None:
        if trainer.current_epoch == 0:
            trainer.train_dataloader.num_workers = 0

    def on_validation_epoch_end(self, trainer: Trainer, _) -> None:
        if trainer.current_epoch == 0:
            trainer.val_dataloaders.num_workers = 0


base_dir = Path(__file__).resolve().parent.parent


def load_conf(conf_file: str = base_dir / "src/rnn/config.yml"):
    with open(conf_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_conf()

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["model_params"]["name"],
    )

    # For reproducibility
    seed_everything(config["train_params"]["manual_seed"], True)

    vae_model = VanillaVAE.load_from_checkpoint(base_dir / "logs/VAE/nan_0.7_fixed/checkpoints/last.ckpt")
    rnn_model = InterpRNN(**config, vae_model=vae_model)

    data = RNNDataModule(
        **config["data_params"],
        pin_memory=config["trainer_params"]["accelerator"] == "gpu",
    )
    data.setup()

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[
            UpdateParams(),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        **config["trainer_params"],
    )

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(rnn_model, datamodule=data)
