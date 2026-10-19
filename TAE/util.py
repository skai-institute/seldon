import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from collections.abc import Mapping
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

import math
from tqdm import tqdm

OmegaConf.register_new_resolver("sum", sum)
OmegaConf.register_new_resolver("len", sum)

def read_transmission_filter(filename):
    filter_data = pd.read_csv(filename, sep=r'\s+', comment='#', header=None, names=['wavelength', 'throughput'])
    wavelength = filter_data['wavelength'].to_numpy()
    if 'jwst' in filename: # wavelength units in microns
        wavelength = wavelength * 1e4
    elif 'total' in filename: # nanomaters for LSST Filters
        wavelength = wavelength * 10

    wavelength = wavelength*1e-4 # Back to microns
    #wavelength * 1e-4 # Convert to Microns
    throughput = filter_data['throughput'].values

    return wavelength, throughput


def map_transmission_filters(dictionary, n_points=None, wave_min=None, wave_max=None):

    filters = {key: read_transmission_filter(filename) for key, filename in dictionary.items()}
    all_waves = np.hstack([value[0] for value in filters.values()])
    if wave_min is None:
        wave_min = all_waves.min()
    if wave_max is None:
        wave_max = all_waves.max()
    if n_points is None:
        n_points = len(np.unique(all_waves))

    wave_grid = np.logspace(np.log10(wave_min), np.log10(wave_max), n_points)
    interpolators = {key: interp1d(*value, fill_value=0.0, bounds_error=False) for key, value in filters.items()}
    filter_embeddings = {key: value(wave_grid) for key, value in interpolators.items()}

    return wave_grid, filter_embeddings

registry = {}

def register_module(cls):
    global registry

    name = cls.__name__
    if name in registry:
        raise KeyError(f"Name {name} already in the registry at {cls.__module__}")
    registry[name] = cls
    return  cls



# Takes a mappable and flattens it into key1.key2...keyN -> mappabe[key1][key2]...[keyN]
def flatten_config(config):
    new_config = type(config)({})
    for key1 in config:
        if isinstance(config[key1], Mapping):
            sub_dict = flatten_config(config[key1])
            for key2 in sub_dict:
                new_config['.'.join((key1, key2))] = sub_dict[key2]
        else:
            new_config[key1] = config[key1]
    return new_config
            
# Takes a config and allows subsitution of key components
def fill_config(config):
    OmegaConf.resolve(config)
    return config

def extract_curves_from_filtered_data(filtered_data, min_points=20, class_filter=None):
    """Extract light curves from the filtered data structure, with optional filtering by class."""
    light_curves = []
    class_names = []

    for class_name, band_dict in tqdm(filtered_data.items()):
        if isinstance(class_filter, str):
            if class_filter is not None and class_name != class_filter:
                continue
        else:
            if class_filter is not None and class_name not in class_filter:
                continue

        if isinstance(band_dict, dict):
            light_curve = []
            for band, curves in band_dict.items():
                if isinstance(curves, int):
                    continue

                for curve in curves:
                    if len(curve) >= min_points:
                        light_curve.append(curve)

            light_curve = pd.concat(light_curve, axis=0).sort_values(
                "days_since_first_observation"
            )
            light_curves.append(light_curve)
            class_names.append(class_name)
        else:
            light_curves.extend(band_dict)
            class_names.extend([class_name]*len(band_dict))

    return light_curves, class_names



def linear_to_log(f_lin):
    return torch.sign(f_lin) * torch.log10(f_lin.abs() + 1.0)

def log_to_linear(f_log):
    return torch.sign(f_log) * (10.0 ** f_log.abs() - 1.0)

def sigma_log_to_linear(s_log, f_log):
    return s_log * (10.0 ** f_log.abs()) * math.log(10.0)
