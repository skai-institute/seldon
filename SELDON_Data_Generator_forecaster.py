import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
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
from functools import partial
import pandas as pd
torch.set_float32_matmul_precision('high')




BAND_IDX_MAP = {0: "g", 1:"r", 2:"i", 3:"z", 4:"y", 5: "u"}
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


def megabatch_inputs(raw_idxs, samples=None, masks=None, bands=None):

    return

def megabatch_outputs(n_samples, times, band_idx=None):

    if band_idx is None:
        band_idx = torch.arange(6, dtype=torch.long)
    times, bands = torch.meshgrid(times, band_idx.to(times.dtype), indexing='ij')
    # axis=0 increments time, axis=1 increments bands
    times = times[None, :, :].expand(n_samples, -1, -1).reshape(n_samples, -1, 1)
    bands = bands[None, :, :].expand(n_samples, -1, -1).reshape(n_samples, -1, 1).to(band_idx.dtype)
    # To undo, reshape outputs to (n_samples, n_times, n_bands)

    return times, bands


def split_batches(batch_size=512, *args):

    length = len(args[0])
    n_chunks = length // batch_size
    for i in zip(map(partial(torch.chunk, chunks=n_chunks), args)):
        yield i

def collate_tensors(tensor_list):

    keys = tensor_list[0].keys()
    output_dict = {}
    for key in keys:
        if isinstance(tensor_list[0][key], torch.Tensor):
            output_dict[key] = torch.cat([section[key] for section in tensor_list], dim=0)
        elif isinstance(tensor_list[0][key], np.ndarray):
            output_dict[key] = np.vstack([section[key] for section in tensor_list])
        elif isinstance(tensor_list[0][key], list):
            output_dict[key] = sum([section[key] for section in tensor_list])
        elif isinstance(tensor_list[0][key], dict):
            output_dict[key] = collate_tensors([section[key] for section in tensor_list])
        else:
            output_dict[key] = [section[key] for section in tensor_list]

    return output_dict

def batch_stream(batch, experiment, N_times=200): # Do this for 1 of N_samples, then just iterate the whole thing N_samples times
    # Returns iterator of encoder inputs, decoder inputs, meta_data

    for i in range(len(batch['lengths'])):

        raw_idx = batch['raw_idx'][i]
        length = torch.tensor([batch['lengths'][i]])
        len_data = len(batch['x_input'][i])
        try:
            redshift = torch.tensor(experiment.data.dataset.meta_data['REDSHIFT_FINAL'][raw_idx][0])
            snid = torch.tensor(experiment.data.dataset.meta_data['SNID'][raw_idx][0])
            class_id = torch.tensor([class_map[batch['class_name'][i]]])
        except TypeError:
            redshift = torch.tensor([-1.0])
            snid =  raw_idx
            class_id = torch.tensor([0])

        time = batch["time_full"][i].detach().cpu().numpy()[:length, 0]
        t_max = np.max(np.abs(time)) * 1.2
        t_min = -t_max
        t_dense = torch.linspace(t_min, t_max, N_times)

        times, bands = megabatch_outputs(length, t_dense) # a time for every length
        times = times # (B, S)
        bands = bands # (B, S)

        progressive_mask = torch.tril(torch.ones((length, len_data), dtype=bool), diagonal=0) # A mask with '1..length' batches and appropriate sequence length
        progressive_mask = progressive_mask # (length x len_data)

        x_input = batch['x_input'][i][None,...].expand(length, -1, -1) # (B, S, E)
        band_idx = batch['band_idx'][i][None, ...].expand(length, -1, -1) # (B, S, E)
        in_sample_mask = progressive_mask # (B, S)

        ones_array = torch.ones((length)).to(int)

        encoder_inputs = {'x_input': x_input, 'band_idx': band_idx, 'in_sample_mask': in_sample_mask}
        decoder_inputs = {'times': times, 'bands': bands}
        meta_data = {'redshift': redshift*ones_array, 'snid':snid*ones_array.to(int), 'N_obs': progressive_mask.sum(1), 'length': length*ones_array.to(int), 'raw_idx': raw_idx*ones_array.to(int), 'class_id':class_id*ones_array.to(int)}

        yield {'encoder_inputs':encoder_inputs, 'decoder_inputs':decoder_inputs, 'meta_data':meta_data}

def batch_stream_limited(batch, experiment, N_times=200, max_length=2048):

    collection = {}
    total_length = 0
    for item in batch_stream(batch, experiment, N_times):
        total_length += item['meta_data']['length'][0]
        if total_length > max_length:
            total_length = 0
            yield collection
            collection = {}
            continue
        if not len(collection):
            collection = item
        else:
            for key in collection:
                #import pdb; pdb.set_trace()
                collection[key] = collate_tensors([collection[key], item[key]])
    if len(collection): # Make sure we get the stragglers
        yield collection
        
def evaluate_batch_collection(collection, experiment):

    model = experiment.model
    device = experiment.device
    with torch.inference_mode():
        encoder_inputs = collection['encoder_inputs']
        decoder_inputs = collection['decoder_inputs']
        N_times = decoder_inputs['times'].shape[1] // 6
        N_samples = len(encoder_inputs['x_input'])

        memory = model.encoder(encoder_inputs['x_input'].to(device),
                               encoder_inputs['band_idx'].to(device),
                               mask=encoder_inputs['in_sample_mask'].to(device),
        )
        latent = model.variational_component(memory)
        z = latent['z_mean']

        out = model.decode(
            z, 
            decoder_inputs['times'].to(device), 
            decoder_inputs['bands'].to(device)
        ).to("cpu")
        out_recon = model.decode(
            z, 
            encoder_inputs['x_input'][..., 0][..., None].to(device), 
            encoder_inputs['band_idx'].to(device)
        ).to("cpu")
        #import pdb; pdb.set_trace()
        latent = {key:value.to("cpu") for key, value in latent.items()}
        out = out.reshape(N_samples, N_times, 6, -1).unbind(2)
        out = dict(zip(map(BAND_IDX_MAP.get, range(6)), out))
        
        #parameters = {
        #    "Basis":model.decoder.last_basis,
        #    "weights": model.decoder.last_weight,
        #    "mu": model.decoder.last_mu,
        #    "sigma": model.decoder.last_sigma,
        #    "redshift": model.decoder.last_redshift,
        #    "band_embedding": model.decoder.last_band_embedding
        #} # this is a large amount of data

        outputs = {"latent": latent, "interpolation": out, "reconstruction":out_recon}#, "parameters":parameters}
        outputs = {'outputs':flatten_dict(outputs)}
        outputs.update(collection)
        outputs = flatten_dict(outputs)

    return outputs

def generate_dataframes(outputs):
    snid_set = torch.unique(outputs['meta_data.snid'], sorted=False)
    input_data_frames = {'meta_data':[], 'input_data':[], 'reconstruction':[], 'latent':[], 'interpolation':[]}
    for snid in snid_set:
        x_input_small_mask = (outputs['meta_data.snid'] == snid) & (outputs['meta_data.N_obs'] == 1)
        x_input_small = outputs['encoder_inputs.x_input'][x_input_small_mask]
        band_idx_small = outputs['encoder_inputs.band_idx'][x_input_small_mask].to(int)
        snid_small = outputs['meta_data.snid'][x_input_small_mask].to(int)
        redshift_small = outputs['meta_data.redshift'][x_input_small_mask]
        raw_idx_small = outputs['meta_data.raw_idx'][x_input_small_mask].to(int)
        class_id_small = outputs['meta_data.class_id'][x_input_small_mask].to(int)
        length_small = outputs['meta_data.length'][x_input_small_mask].to(int)
        x_input_small = x_input_small[:, :length_small[0]]
        band_idx_small = band_idx_small[:, :length_small[0]]
        if x_input_small.shape[-1] == 4:
            time, flux, flux_err, detection_flag = x_input_small.unbind(2)
            detection_flag = detection_flag.to(bool)
        else:
            time, flux, flux_err = x_input_small.unbind(2)
    
        meta_dataframe = pd.DataFrame(
            {
                'snid': snid_small,
                'raw_idx': raw_idx_small,
                'class_id': class_id_small,
                'length': length_small,
                'redshift': redshift_small,
            }
        ).set_index('snid')
        if x_input_small.shape[-1] == 4:
            light_curve_dataframe = pd.DataFrame(
                {
                    'snid' : snid_small.expand(time.reshape(-1).shape[0]),
                    'scaled_time': time.reshape(-1),
                    'scaled_flux': flux.reshape(-1),
                    'scaled_flux_err': flux_err.reshape(-1),
                    'detection_flag': detection_flag.reshape(-1),
                    'band_idx': band_idx_small.reshape(-1),
                }
            ).reset_index().rename(columns={'index':'sequence_index'}).set_index(['snid', 'sequence_index'])
        else:
            light_curve_dataframe = pd.DataFrame(
                {
                    'snid' : snid_small.expand(time.reshape(-1).shape[0]),
                    'scaled_time': time.reshape(-1),
                    'scaled_flux': flux.reshape(-1),
                    'scaled_flux_err': flux_err.reshape(-1),
                    'band_idx': band_idx_small.reshape(-1),
                }
            ).reset_index().rename(columns={'index':'sequence_index'}).set_index(['snid', 'sequence_index'])
 
        snid_mask = (outputs['meta_data.snid'] == snid)
        reconstructions = outputs['outputs.reconstruction'][snid_mask, :length_small[0], 0]
        seq_index = torch.arange(length_small[0])[None, :].expand(len(reconstructions), -1)
        reconstruction_snid = outputs['meta_data.snid'][snid_mask][None,:].expand(*seq_index.shape).to(int)
        reconstruction_n_obs = outputs['meta_data.N_obs'][snid_mask][:, None].expand(*seq_index.shape).to(int)
        reconstruction_dataframe = pd.DataFrame(
            {
                'snid': reconstruction_snid.reshape(-1),
                'sequence_index': seq_index.reshape(-1),
                'N_obs': reconstruction_n_obs.reshape(-1),
                'reconstruction': reconstructions.reshape(-1)
            },
        ).set_index(['snid','N_obs','sequence_index'])
    
        latent_keys = [key for key in outputs if 'latent' in key]
        latent_variables = {key.split('.')[-1]+f'_{i}': z[snid_mask] for key in latent_keys for i, z in enumerate(outputs[key].unbind(-1))}
        latent_variables.update(
            {
                'snid': outputs['meta_data.snid'][snid_mask].to(int),
                'N_obs': outputs['meta_data.N_obs'][snid_mask].to(int)
            }
        )
        latent_dataframe = pd.DataFrame(latent_variables).set_index(['snid', 'N_obs'])
    
        interpolation_keys = [key for key in outputs if 'interpolation' in key]
        interpolation_variables = {key.split('.')[-1]: outputs[key][snid_mask, :, 0] for key in interpolation_keys}
        m, n = interpolation_variables['g'].shape[:2]
        interpolation_variables.update(
            {
                'time':outputs['decoder_inputs.times'][snid_mask][:, ::6, 0],
                'snid': outputs['meta_data.snid'][snid_mask].to(int)[:, None].expand(m, n),
                'N_obs': outputs['meta_data.N_obs'][snid_mask].to(int)[:, None].expand(m, n),
                'sequence_index': torch.arange(N_times)[None, :].expand(m, n)
            }
        )
        interpolation_variables = {key: value.reshape(-1) for key, value in interpolation_variables.items()}
        interpolation_dataframe = pd.DataFrame(interpolation_variables).set_index(['snid', 'N_obs', 'sequence_index'])
    
        input_data_frames['meta_data'].append(meta_dataframe)
        input_data_frames['input_data'].append(light_curve_dataframe)
        input_data_frames['reconstruction'].append(reconstruction_dataframe)
        input_data_frames['latent'].append(latent_dataframe)
        input_data_frames['interpolation'].append(interpolation_dataframe)
    
    input_data_frames['meta_data'] = pd.concat(input_data_frames['meta_data'])
    input_data_frames['input_data'] = pd.concat(input_data_frames['input_data'])
    input_data_frames['reconstruction'] = pd.concat(input_data_frames['reconstruction'])
    input_data_frames['latent'] = pd.concat(input_data_frames['latent'])
    input_data_frames['interpolation'] = pd.concat(input_data_frames['interpolation'])

    return input_data_frames


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '32'

    config_path_str = "/work/hdd/bejy/mkrafcz2/checkpoints/version_5"

    parser = argparse.ArgumentParser()
    _ = parser.add_argument('--config', type=str, nargs="+", required=True, default=config_path_str)
    args = parser.parse_args()

    N_samples = 1
    N_times = 200

    config_path_strs = cast(list[str], args.config)
    config_paths = list(map(Path, config_path_strs))

    # Load the data for the experiment, assumed the first config has the correct information
    first_config = config_paths[0]

    first_config_file = first_config / 'hparams.yaml'

    data_cfg = OmegaConf.load(first_config_file)
    data_cfg['dataset']['config']['num_workers'] = 1
    data_cfg['dataset']['config']['augmentations'] = ['FullSample']
    data_cfg['dataset']['config']['batch_size'] = 512

    data_cfg = fill_config(data_cfg)
    data_config = OmegaConf.to_container(data_cfg)

    data = get_dataset(data_config["dataset"]["name"], data_config["dataset"]["config"])
    #data.setup()

    count_info = datamodule_counts(data)
    num_batches = count_info['val'][0]['dataset_len']

    for config_path in config_paths:
        config_file = config_path / 'hparams.yaml'
        weights_file = list(config_path.glob('checkpoints/epoch*'))[0]

        cfg = OmegaConf.load(config_file)
        cfg['dataset']['config']['num_workers'] = 1
        data_cfg['dataset']['config']['augmentations'] = ['FullSample']
        data_cfg['dataset']['config']['batch_size'] = 512

        cfg = fill_config(cfg)
        config = OmegaConf.to_container(cfg)

        # === Set seed ===
        seed = cfg["logging_params"]["manual_seed"]
        pl.seed_everything(seed)

        model = get_module(cfg["model_params"]["name"], config["model_params"]["config"])
        model.requires_grad_(False)
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

        data_loader = experiment.data.test_dataloader()
        class_names = sorted(list(set(experiment.data.dataset.class_names)))
        class_map = dict(zip(class_names, range(len(class_names))))
        class_map.update(experiment.data.flux_stats)
        print(class_map)
        with open(config_path / 'class_map.txt', 'w') as f:
            f.write(str(class_map))
        with open(config_path/"flux_stats.txt", 'w') as f:
            f.write(str(experiment.data.flux_stats))

        batch_data = []
        device = experiment.device

        for k in tqdm(range(N_samples)):
            all_outputs = {}
            batch_stem = f"{cfg['logging_params']['name']}_{k}"
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Processing {config_path.name}', leave=False):
                all_outputs[str(i)] = {}
                for j, collection in enumerate(batch_stream_limited(batch, experiment, N_times=N_times, max_length=2048)):
                    #outputs = evaluate_batch_collection(collection, experiment)
                    outputs = generate_dataframes(evaluate_batch_collection(collection, experiment))
                    if i == 0:
                        dataframes = {key: [outputs[key]] for key in outputs}
                    else:
                        for key in dataframes:
                            dataframes[key].append(outputs[key])
            dataframes = {key: pd.concat(dataframes[key]) for key in dataframes}
            for key in dataframes:
                dataframes[key].to_parquet(path=config_path/f"{batch_stem}.{key}.{k}.parquet")
            del dataframes
            del outputs
                    #if j == 0:
                    #    all_outputs[str(i)] = outputs
                    #else:
                    #    all_outputs[str(i)] = collate_tensors([all_outputs[str(i)], outputs])
            #all_ouputs = flatten_dict(all_outputs)
            #save(all_outputs, config_path/f"{batch_stem}.reconstruction_data.safetensors")


        '''
        for i, batch in enumerate(tqdm(data_loader, total=num_batches, desc=f"Processing {config_path.name}")):
            batch_cpu = batch
            if batch['lengths'][0] < 50:
                continue
            length = batch['lengths'][0]

            batch_outputs = []
            
            raw_idx = batch['raw_idx'][0] # index of the object in the batch, can resample from the LC preprocessor
            batches_resampled = [experiment.data.dataset.legacy_get(raw_idx) for _ in range(N_samples)]
            batches_resampled = enhanced_collate_fn(batches_resampled)
            old_in_sample_mask = batches_resampled['in_sample_mask'].clone()
            len_data = len(old_in_sample_mask[0])
            redshift = torch.tensor(experiment.data.dataset.meta_data['REDSHIFT_FINAL'][raw_idx])
            snid = torch.tensor(experiment.data.dataset.meta_data['SNID'][raw_idx])

            #BAND_IDX_MAP = {"g": 0, "r": 1, "i": 2, "z": 3, "y": 4, "u": 5}
            BAND_IDX_MAP = {0: "g", 1:"r", 2:"i", 3:"z", 4:"y", 5: "u"}

            time = batches_resampled["time_full"][0].detach().cpu().numpy()[:length, 0]
            t_max = np.max(np.abs(time)) * 1.2
            t_min = -t_max
            t_dense = torch.linspace(t_min, t_max, N_times)
            object_outputs = {}
            batches_resampled = {key: to_cuda(value, device='cuda:0') if isinstance(value, torch.Tensor) else value for key, value in batches_resampled.items()}

            #times = t_dense[None, :, None].expand(N_samples, -1, 1)
            #bands = torch.arange(6, dtype=torch.long)[None, None, :, None]
            #bands = bands.expand(N_samples, N_times, -1, -1).reshape(N_samples, 6*N_times, -1).to(device)
            #times = times[:, :, None, :].expand(-1, -1, 6, -1).reshape(N_samples, 6*N_times, -1).to(device)
            times, bands = megabatch_outputs(N_samples, t_dense)
            times = times[..., None].to(device)
            bands = bands[..., None].to(device)

            progressive_mask = torch.tril(torch.ones((length-1, len_data), dtype=bool), diagonal=1)[..., None].to(device) # A mask with '1..length' batches and appropriate sequence length
            progressive_mask = progressive_mask[None, ...].expand(N_samples, -1, -1, -1)

            for j in tqdm(range(1, length)):
                in_sample_mask = progressive_mask[:, j-1]
                with torch.inference_mode():
                    latent = model.encode(
                                batches_resampled['x_input'],
                                batches_resampled['band_idx'],
                                mask=in_sample_mask[..., 0],
                            )

                    #import pdb;pdb.set_trace()
                    out = model.decode(latent, times[..., 0], bands[..., 0])
                out = out.reshape(N_samples, N_times, 6, -1).unbind(2)
                out = dict(zip(map(BAND_IDX_MAP.get, range(6)), out))

                outputs = {"latent": latent, "reconstruction": out}#, 'redshift':redshift, 'snid':snid}
                outputs = flatten_dict(outputs)

                object_outputs[str(j)] = outputs

            object_outputs = flatten_dict(object_outputs)
            batches_resampled['class_name'] = torch.tensor([class_map[i] for i in batches_resampled['class_name']], device='cuda:0')
            
            
            all_outputs = {'batch': batches_resampled, 'outputs': object_outputs, 't_grid': t_dense, "redshift": redshift, "snid": snid}
            all_outputs = flatten_dict(all_outputs)
            batch_stem = f"{cfg['logging_params']['name']}_{i}"
            save(all_outputs, config_path/f"{batch_stem}.reconstruction_data.safetensors")
            '''
            
