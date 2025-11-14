import pytorch_lightning as pl
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
cudnn.deterministic = True
cudnn.benchmark = True
import argparse
from typing import cast
from TAE.visualizations.percentile_forecasting import percentile_forecast
import json
from datetime import datetime
from TAE.datasets.ephemeral import save, load
from tqdm import tqdm
from TAE.datasets.utils import datamodule_counts, to_cuda

def flatten_dict(dictionary):

    new_dict = {}
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            flat_dict = flatten_dict(dictionary[key])
            flat_dict = {'.'.join((key, key2)): flat_dict[key2] for key2 in flat_dict}
            new_dict.update(flat_dict)
        elif isinstance(dictionary[key], torch.Tensor):
            new_dict[key] = dictionary[key]
        elif isinstance(dictionary[key], list):
            if isinstance(dictionary[key][0], torch.Tensor):
                new_dict[key] = dictionary[key]
    return new_dict

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '32'

    config_path_str = "/work/hdd/bejy/mkrafcz2/checkpoints/version_5"

    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--config', type=str, nargs="+", required=True, default=config_path_str)
    args = parser.parse_args()

    config_path_strs = cast(list[str], args.config)
    config_paths = list(map(Path, config_path_strs))

    # Load the data for the experiment, assumed the first config has the correct information
    first_config = config_paths[0]

    first_config_file = first_config / 'hparams.yaml'

    data_cfg = OmegaConf.load(first_config_file)
    data_cfg['dataset']['config']['num_workers'] = 1
    data_cfg['dataset']['config']['augmentations'] = ['FullSample']

    data_cfg = fill_config(data_cfg)
    data_config = OmegaConf.to_container(data_cfg)

    data = get_dataset(data_config["dataset"]["name"], data_config["dataset"]["config"])
    data.setup()
    #data.dataset.cuda()

    count_info = datamodule_counts(data)
    num_batches = count_info['val'][0]['dataset_len']

    for config_path in config_paths:
        config_file = config_path / 'hparams.yaml'
        weights_file = list(config_path.glob('checkpoints/epoch*'))[0]

        cfg = OmegaConf.load(config_file)
        cfg['dataset']['config']['num_workers'] = 1
        data_cfg['dataset']['config']['augmentations'] = ['FullSample']

        cfg = fill_config(cfg)
        config = OmegaConf.to_container(cfg)

        # === Set seed ===
        seed = cfg["logging_params"]["manual_seed"]
        pl.seed_everything(seed)

        model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])
        model.cuda()

        # === Load experiment wrapper ===
        experiment = get_experiment(
            cfg["exp_params"]["name"],
            {"model": model, "data": data, **config["exp_params"]["config"]},
        )

        # for over models
        weights_file = list(config_path.glob('checkpoints/epoch*'))[0]
        experiment.load_from_checkpoint(checkpoint_path=weights_file,
                                        model=model,
                                        data=data,
                                        params=cfg["exp_params"]["config"]["params"]);

        _ = experiment.cuda()
        _ = experiment.eval()
        model.eval()

        data_loader = experiment.data.train_dataloader()
        class_names = set(experiment.data.dataset.class_names)
        class_map = dict(zip(class_names, range(len(class_names))))
        print(class_map)
        with open(config_path / 'class_map.txt', 'w') as f:
            f.write(str(class_map))
        with open(config_path/"flux_stats.txt", 'w') as f:
            f.write(str(experiment.data.flux_stats))

        for i, batch in enumerate(tqdm(data_loader, total=num_batches, desc=f"Processing {config_path.name}")):
            #to_cuda(batch, device='cuda:0')
            batch_cpu = batch
            batch = {key: to_cuda(value, device='cuda:0') if isinstance(value, torch.Tensor) else value for key, value in batch.items() }

            with torch.inference_mode():
                outputs = experiment.validation_step(batch, i)

            outputs['batch']['class_name'] = torch.tensor([class_map[i] for i in outputs['batch']['class_name']], device='cuda:0')
            redshift = torch.tensor([experiment.data.dataset.meta_data['REDSHIFT_FINAL'][raw_idx] for raw_idx in outputs['batch']['raw_idx']], device='cuda:0')
            snid = torch.tensor([experiment.data.dataset.meta_data['SNID'][raw_idx] for raw_idx in outputs['batch']['raw_idx']], device='cuda:0')

            outputs['batch']['redshift'] = redshift
            outputs['batch']['snid'] = snid
            outputs = flatten_dict(outputs)

            batch_stem = f"{cfg['logging_params']['name']}_{i}"

            save(outputs, config_path/f"{batch_stem}_outputs.safetensors")
            #save(batch, config_path/f"{batch_stem}_batch.safetensors")

            #resid = percentile_forecast(experiment, batch)
            #resid = flatten_dict(resid)

            #print(resid.keys)
            #save(resid, f"{batch_stem}_resid_{i}.safetensors")
        #if self.metrics is not None:
        #    metrics = self.metrics(outputs[0])
        #    for metric in metrics:
        #        #self.log(metric, metrics[metric])
        #        values = metrics[metric]
        #        if 'Max' in metric or "FE" in metric:
        #            values = torch.log10(values)
        #        if torch.any(torch.isfinite(values)):
        #            self.logger.experiment.add_histogram(metric, values, self.current_epoch)

