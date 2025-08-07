import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import math
from tqdm import tqdm



registry = {}

def register_module(cls):
    global registry

    name = cls.__name__
    if name in registry:
        raise KeyError(f"Name {name} already in the registry at {cls.__module__}")
    registry[name] = cls
    return  cls

# Minimal fill_config that just passes the config through
def fill_config(config):
    # Return the config without attempting any modifications
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
