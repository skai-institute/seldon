import math
import dataclasses as _dc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from TAE.util import log_to_linear, sigma_log_to_linear, linear_to_log, register_module


# ---------------------------------------------------------------------
#  Shared helpers & constants
# ---------------------------------------------------------------------

FLUX_FLOOR = 0.01
BAND_IDX_MAP = {"g":0, "r":1, "i":2, "z":3, "y":4, "u":5}
BAND_COL = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
]
IDX2NAME = {v:k for k, v in BAND_IDX_MAP.items()}

@register_module
class Metrics:

    def __init__(self, metrics, log_flux=True):

        self.metrics = metrics
        self.log_flux = log_flux
        self.metric_dict = {}

    def __call__(self, outputs):

        for metric in self.metrics:
            self.metric_dict.update(getattr(self, metric)(outputs))
        return self.metric_dict

    def __getitem__(self, metric):
        return self.metric_dict[metric]
    

    def fractional_errors(self, output):

        batch = output['batch']
        pred = output['out']['reconstructed'][..., 0]
        true = batch['flux_full'][..., 0]
        mask = batch['pad_mask_full']
        if self.log_flux:
            scale = batch['flux_norm_info'][0][-1]
            pred = (pred + 0.5)*scale
            true = (true + 0.5)*scale
            pred = log_to_linear(pred)
            true = log_to_linear(true)

        denom = true.abs().clamp(FLUX_FLOOR)
        fe = (pred - true).abs() / denom * mask
        counts = mask.sum(1).clamp_min(1)
        mean_fe = fe.sum(1)/counts
        max_fe = (fe + (~mask)*(-1e9)).max(1)[0]
        return {'MaxFE': max_fe, 'MeanFE': mean_fe}
    

    def z_scores(self, output):

        batch = output['batch']
        pred = output['out']['reconstructed'][..., 0]
        true = batch['flux_full'][..., 0]
        mask = batch['pad_mask_full']
        sigma = batch['flux_err_full'][..., 0]
        if self.log_flux:
            scale = batch['flux_norm_info'][0][-1]
            pred = (pred + 0.5)*scale
            true = (true + 0.5)*scale
            
            pred = log_to_linear(pred)
            true = log_to_linear(true)
            sigma = sigma * scale * np.log(10) * (torch.abs(pred) + 1)

        z = (pred - true) / (
            sigma) * mask# + (~mask) * 1e9)
        counts = mask.sum(1).clamp_min(1)
        
        z_mean = z.sum(1) / counts
        z_max = (z.abs()).max(1)[0]

        return {'MeanZ': z_mean, 'MaxZ': z_max}

    def nrmse(self, output):

        batch = output['batch']
        pred = output['out']['reconstructed'][..., 0]
        true = batch['flux_full'][..., 0]
        mask = batch['pad_mask_full']
        if self.log_flux:
            scale = batch['flux_norm_info'][0][-1]
            pred = (pred + 0.5)*scale
            true = (true + 0.5)*scale
            pred = log_to_linear(pred)
            true = log_to_linear(true)

        counts = mask.sum(1).clamp_min(1)
        _, f_pk = self._peak_stats(batch["time_full"][..., 0], true)
        nrmse = torch.sqrt(((pred - true) ** 2).sum(1) / counts) \
                / f_pk.clamp_min(1e-6)
        return {'NRMSE':nrmse}
    

    def _peak_stats(self, t, f):
        idx_pk = torch.argmax(f, dim = 1, keepdim = True)
        t_pk = torch.gather(t, 1, idx_pk)
        f_pk = torch.gather(f, 1, idx_pk)
        return t_pk, f_pk



class MetricsEvaluator:
    def __init__(self, model, batch, *, device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
        self.mask = self.batch['in_sample_mask'] #self.batch["mask"].shape == (B, L)
        self.out = self.model(self.batch)
        self.pred_log = log_to_linear(self.out["reconstructed"])
        #print(self.mask.shape, self.batch['flux_full'].shape)
        self.true_lin = log_to_linear(self.batch['flux_full'][:, :, 0])*self.mask
        self.pred_lin = log_to_linear(self.pred_log[:, :, 0]) * self.mask
        
    def fractional_errors(self):
        denom = self.true_lin.abs().clamp_min(FLUX_FLOOR)
        fe = (self.pred_lin - self.true_lin).abs()/denom * self.mask
        counts = self.mask.sum(1).clamp_min(1)
        mean_fe = fe.sum(1)/counts
        max_fe = (fe + (~self.mask)*(-1e9)).max(1).values
        return mean_fe.squeeze(-1), max_fe.squeeze(-1)
    
    def z_scores(self):
        z = (self.pred_log[:, :, 0] - self.batch["flux_full"][:, :, 0]) / (
            self.batch["flux_err_full"][:, :, 0] * self.mask + (~self.mask) * 1e9)
        counts = self.mask.sum(1).clamp_min(1)
        z_mean = (z.sum(1) / counts).squeeze(-1)
        z_max = (z.abs() + (~self.mask) * (-1e9)).max(1).values.squeeze(-1)
        return z_mean, z_max

    def nrmse(self):
        counts = self.mask.sum(1).clamp_min(1)
        _, f_pk = self._peak_stats(self.batch["time_full"][:, :, 0], self.true_lin)
        nrmse = torch.sqrt(((self.pred_lin - self.true_lin) ** 2).sum(1) / counts) \
                / f_pk.squeeze(-1).clamp_min(1e-6)
        return nrmse.squeeze(-1)
    
    def _peak_stats(self,t,f):
        idx_pk = torch.argmax(f, dim = 1, keepdim = True)
        t_pk = torch.gather(t, dim = 1, idx = idx_pk)
        f_pk = torch.gather(f, 1, idx_pk)
        return t_pk, f_pk
    
    def all_metrics(self):
        fe_mean, fe_max = self.fractional_errors()
        z_mean, z_max = self.z_scores()
        nrmse = self.nrmse()
        return {
            "MeanFE": fe_mean,
            "MaxFE": fe_max,
            "meanZ": z_mean,
            "max|Z|": z_max,
            "NRMSE": nrmse,
        }
        
def select_indices(vec, frac = 0.1):
    n = max(1, int(len(vec)*frac))
    idx_sorted = torch.argsort(vec)
    # find the starting index of the middle n elements in a sorted version of the array vec
    mid = len(vec)//2-n//2
    return {
        "best": idx_sorted[:n].tolist(),
        "median": idx_sorted[mid:mid + n].tolist(),
        "worst": idx_sorted[-n:].tolist(),
    }