import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import logging
import sys
import hydra
import pytorch_lightning as pl
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.profiler import PyTorchProfiler

from TAE.datasets.dataset_factory import get_module as get_dataset
from TAE.experiments.factory import get_module as get_experiment
from TAE.models.factory import get_module
from TAE.util import fill_config
from TAE.datasets import dataset_factory

import torch


torch.set_float32_matmul_precision('high')
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
torch.set_num_interop_threads(int(os.environ['OMP_NUM_THREADS']))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True
#torch._dynamo.config.capture_scalar_outputs = True
#torch._dynamo.config.recompile_limit = 12


logger = logging.getLogger(__name__)


@hydra.main(config_path="TAE/configs", config_name="light_curve_autoencoder", version_base="1.1")
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
        save_weights_only=False,
        filename="{epoch}-{val_loss:.2f}",
        verbose=True,
        save_last=False,
        every_n_epochs=1,
    )

    # === Load model ===
    model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])
    #model.compile(mode="reduce-overhead")

    # === Prepare datamodule manually ===
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    # === Load experiment wrapper ===
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )

    if 'resume_from_checkpoint' in cfg['trainer_params']:
        if cfg['trainer_params']['resume_from_checkpoint'] is not None:
            chkpt = cfg['trainer_params']['resume_from_checkpoint']
            experiment.load_from_checkpoint(checkpoint_path=chkpt,
                        model=model,
                        data=data,
                        params=cfg["exp_params"]["config"]["params"]);
            cfg['trainer_params']['resume_from_checkpoint'] = None

    # === Log and run ===
    tb_logger.log_hyperparams(params=cfg)
    logger.info(f"{tb_logger.save_dir}")

    #profiler = PyTorchProfiler(filename="trace_name.prof")
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
