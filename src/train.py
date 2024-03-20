import os
from pathlib import Path

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ai import VAEDataModule, VAEXperiment, VanillaVAE


def load_conf(conf_file: str = Path(__file__).resolve().parent / "ai/config.yml"):
    with open(conf_file, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


if __name__ == "__main__":
    config = load_conf()

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["model_params"]["name"],
    )

    # For reproducibility
    seed_everything(config["exp_params"]["manual_seed"], True)

    model = VanillaVAE(**config["model_params"])
    experiment = VAEXperiment(model, **config)

    data = VAEDataModule(
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
    trainer.fit(experiment, datamodule=data)
