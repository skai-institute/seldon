import logging
import sys

import hydra
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from TAE.datasets.dataset_factory import get_module as get_dataset

from TAE.experiments.factory import get_module as get_experiment
from TAE.models.factory import get_module
from TAE.util import fill_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="human_CKNNAE")
def main(cfg: DictConfig) -> None:
    cfg = fill_config(cfg)
    print(cfg)
    config = OmegaConf.to_container(cfg)
    # logger.info("Found", torch.cuda.device_count(), "devices")

    seed = cfg["logging_params"]["manual_seed"]
    pl.seed_everything(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


    tb_logger = TensorBoardLogger(
        save_dir=cfg["logging_params"]["save_dir"],
        name=cfg["logging_params"]["name"],
    )
    

    checkpoint_callback = ModelCheckpoint(
        # dirpath=f"{tb_logger.save_dir}",
        monitor="val_evaluation",
        mode="min",
        save_weights_only=True,
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,
        save_last=True,
        every_n_epochs=1,
    )

    # get model from yaml
    model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])  # type: ignore
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])  # type: ignore
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )
    with open_dict(cfg):
        preprocessing_data = data.prepare_data()
        if preprocessing_data is not None:
            preprocessing_data = {key: list(map(float, value)) for key, value in data.prepare_data().items()}
            cfg['dataset']['config'].update(preprocessing_data)
            cfg['dataset']['config']['mode'] = 'eval'
            data.mode = 'eval'
    
    tb_logger.log_hyperparams(params=cfg)

    logger.info(f"{tb_logger.save_dir}")
    # TODO: set val check properly
    runner = Trainer(
        # weights_save_path=f"{tb_logger.save_dir}",
        enable_checkpointing=True,
        callbacks=checkpoint_callback,
        # checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        # val_check_interval=0.0,
        # num_sanity_val_steps=1024,
        **cfg["trainer_params"],
    )

    logger.info(f"======= Training {cfg['model_params']['name']} =======")
    runner.fit(experiment, data)
    # runner.test(experiment, data, ckpt_path="best")


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./")
    main()
