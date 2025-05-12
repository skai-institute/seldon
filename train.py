import logging
import sys
import hydra
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from TAE.datasets.dataset_factory import get_module as get_dataset
from TAE.experiments.factory import get_module as get_experiment
from TAE.models.factory import get_module
from TAE.util import fill_config
from TAE.datasets import dataset_factory
import os
os.environ['HYDRA_FULL_ERROR'] = '1'

logger = logging.getLogger(__name__)

@hydra.main(config_path="TAE/configs", config_name="light_curve_autoencoder")
def main(cfg: DictConfig) -> None:
    cfg = fill_config(cfg)
    print(OmegaConf.to_yaml(cfg))  # Optional: print config for verification
    config = OmegaConf.to_container(cfg)

    # === Set seed ===
    seed = cfg["logging_params"]["manual_seed"]
    pl.seed_everything(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True

    # === Logger and checkpoint ===
    tb_logger = TensorBoardLogger(
        save_dir=cfg["logging_params"]["save_dir"],
        name=cfg["logging_params"]["name"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,
        save_last=True,
        every_n_epochs=1,
    )

    # === Load model ===
    model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])


    # === Prepare datamodule manually ===
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    # === Load experiment wrapper ===
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )

    # === Log and run ===
    tb_logger.log_hyperparams(params=cfg)
    logger.info(f"{tb_logger.save_dir}")

    runner = Trainer(
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        num_sanity_val_steps=0,
        **cfg["trainer_params"],
    )

    logger.info(f"======= Training {cfg['model_params']['name']} =======")
    runner.fit(experiment, data)

if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./")
    main()
