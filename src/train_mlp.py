import os
from pathlib import Path

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlp import InterpMLP, MLPDataModule
from vae import VanillaVAE

base_dir = Path(__file__).resolve().parent.parent


def load_conf(conf_file: str = base_dir / "src/mlp/config.yml"):
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

    vae_model = VanillaVAE.load_from_checkpoint(base_dir / "logs/VAE/nan_0.7_latent_5/checkpoints/last.ckpt")
    mlp_model = InterpMLP(**config, vae_model=vae_model)

    data = MLPDataModule(
        **config["data_params"],
        pin_memory=config["trainer_params"]["accelerator"] == "gpu",
    )
    data.setup()

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[
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

    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(mlp_model, datamodule=data)