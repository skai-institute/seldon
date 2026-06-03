import lightning.pytorch as pl
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from TAE.datasets.dataset_factory import get_module as get_dataset
from TAE.experiments.factory import get_module as get_experiment
from TAE.models.factory import get_module
from TAE.util import fill_config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
import torch
import os
import sys

version = 25
logdir = Path('/scratch/jackob/logs/BandAwareGRUVAE')
config_path = logdir / f'version_{version}'

def load_model(config_path, device='cuda', load_weights=True):
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals([np.dtypes.Float32DType])
    torch.serialization.add_safe_globals([np.dtypes.Float64DType])
    torch.serialization.add_safe_globals([np.dtype])
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    config_path = Path(config_path)


    config_file = config_path / 'hparams.yaml'
    weights_file = list(config_path.glob('checkpoints/epoch*'))[0]

    cfg = OmegaConf.load(config_file)
    cfg['dataset']['config']['num_workers'] = 1
    cfg['dataset']['config']['batch_size'] = 256

    cfg = fill_config(cfg)
    config = OmegaConf.to_container(cfg)

    # === Set seed ===
    seed = cfg["logging_params"]["manual_seed"]
    pl.seed_everything(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])
    data.setup()

    # === Load experiment wrapper ===
    experiment = get_experiment(
        cfg["exp_params"]["name"],
        {"model": model, "data": data, **config["exp_params"]["config"]},
    )
    #data.setup()

    if load_weights:
        weights_file = list(config_path.glob('checkpoints/epoch*'))[0]
        #experiment.load_from_checkpoint(checkpoint_path=weights_file)
        try:
            experiment.load_from_checkpoint(checkpoint_path=weights_file,
                                            model=model,
                                            data=data,
                                            weights_only=None,
                                            params=cfg["exp_params"]["config"]["params"], strict=True);
        except Exception as e:
            print(e) # Force Manual Loading Because it is stupid
            experiment = get_experiment(
                cfg["exp_params"]["name"],
                {"model": model, "data": data, **config["exp_params"]["config"]},
            )
            #experiment.model.energy_model = experiment.energy_model
            experiment.model.energy_model = energy_model
            weights = torch.load(weights_file, map_location=torch.device(device))['state_dict']
            #experiment.load_state_dict(weights)
            for name, parameter in experiment.named_parameters():
                if 'energy_model' in name:
                    print(name)
                parameter.data = weights[name]
    if device == 'cuda':
        model.cuda()
        experiment.cuda()

    data_loader = experiment.data.val_dataloader()

    print("Warming up model...")
    batch = next(iter(data_loader))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(experiment.device) 
    outputs = experiment(batch)

    model.eval()
    experiment.eval()

    return experiment