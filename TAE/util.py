import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# Minimal fill_config that just passes the config through
def fill_config(config):
    # Return the config without attempting any modifications
    return config


def extract_curves_from_filtered_data(filtered_data, min_points=20, class_filter=None):
    """Extract light curves from the filtered data structure, with optional filtering by class."""
    light_curves = []

    for class_name, band_dict in filtered_data.items():
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
        else:
            light_curves.extend(band_dict)

    return light_curves
