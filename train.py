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
from TAE.util import fill_config, extract_curves_from_filtered_data
import pickle
from TAE.datasets import dataset_factory

logger = logging.getLogger(__name__)

@hydra.main(config_path="TAE/configs", config_name="light_curve_autoencoder")
def main(cfg: DictConfig) -> None:
    cfg = fill_config(cfg)
    print(OmegaConf.to_yaml(cfg))  # Optional: print config for verification
    config = OmegaConf.to_container(cfg, resolve=True)

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

    # === Load and extract light curves ===
    with open(cfg["dataset"]["config"]["lightcurve_path"], "rb") as f:
        filtered_data = pickle.load(f)

    light_curves = extract_curves_from_filtered_data(
        filtered_data,
        min_points=cfg["dataset"]["config"]["min_points"],
        class_filter=cfg["dataset"]["config"]["class_filter"]
    )

    # === Prepare datamodule manually ===
    dataset_class = dataset_factory.MODULES[cfg["dataset"]["name"]]

    data = dataset_class(
        full_light_curves=light_curves,
        normalization=cfg["dataset"]["config"]["normalization"],
        min_points=cfg["dataset"]["config"]["min_points"],
        augmentation=cfg["dataset"]["config"].get("augmentation", "mixed"),
        num_aug_per_curve=cfg["dataset"]["config"].get("num_aug_per_curve", 3),
        batch_size=cfg["dataset"]["config"]["batch_size"],
        num_workers=cfg["dataset"]["config"]["num_workers"]
    )



    # === Load experiment wrapper ===
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )

    # === Log and run ===
   # tb_logger.log_hyperparams(params=cfg)

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
