import torch
import matplotlib.pyplot as plt
import math

import numpy as np
from TAE.datasets.utils import BAND_IDX_MAP

def linear_to_log(f_lin):
    return np.sign(f_lin) * np.log10(np.absolute(f_lin) + 1.0)

def log_to_linear(f_log):
    return np.sign(f_log) * (10.0 ** np.absolute(f_log) - 1.0)

def sigma_log_to_linear(s_log, f_lin):
    #return s_log * (10.0 ** np.absolute(f_log)) * math.log(10.0)
    return s_log * (np.absolute(f_lin) + 1.0) * np.log(10.0)

#BAND_COL = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]

BAND_COL = ["#2da328ff", "#e41a1c", "#ff7f00", "#333333", "#984ea3", "#377eb8"] + [None]*len(BAND_IDX_MAP)
IDX2NAME = dict(zip(BAND_IDX_MAP.values(), BAND_IDX_MAP.keys()))

#BAND_COL = ["#4daf4a", "#e41a1c", "#ff7f00", "#333333", "#984ea3", "#377eb8"]
#IDX2NAME = {0:"g", 1:"r", 2:"i", 3:"z", 4:"y", 5:"u"}
TIME_SCALE = 1.0
#BAND_IDX_MAP = {"g":0, "r":1, "i":2, "z":3, "y":4, "u":5}


def plot_full_multiband_dense_plot(experiment, batch, idx, device, ax=None, dense_pts=300, inverse_scale_reconstruction=True):
    torch.compiler.cudagraph_mark_step_begin()
    #BAND_COL = ["#2da328ff", "#e41a1c", "#ff7f00", "#333333", "#984ea3", "#377eb8"] + [None]*len(experiment.data.band_index_map)
    #IDX2NAME = dict(zip(experiment.data.band_index_map.values(), experiment.data.band_index_map.keys()))
    BAND_COL = experiment.data.band_index_to_colors
    IDX2NAME = experiment.data.index_band_map
    model = experiment.model
    if ax is None:
        fig = plt.figure()
    else:
        plt.sca(ax)
        fig = plt.gcf()
    length = batch["lengths"][idx].detach().cpu().numpy()
    flux = batch["flux_full"][idx].detach().cpu().numpy()[:length, 0]
    flux_err = batch["flux_err_full"][idx].detach().cpu().numpy()[:length, 0]
    time = batch["time_full"][idx].detach().cpu().numpy()[:length, 0]
    band = batch["band_idx_full"][idx].detach().cpu().numpy()[:length, 0]
    sample_mask = batch["in_sample_mask"][idx].detach().cpu().numpy()[:length]
    class_name = batch['class_name'][idx]
    t_max = np.max(np.abs(time)) * 1.2
    t_min = -t_max
    t_dense = torch.linspace(t_min, t_max, dense_pts)

    flux, flux_err = experiment.data.dataset.inverse_transform(flux, flux_err)
    for j in sorted(set(band.ravel())):
        bands = torch.ones(dense_pts, dtype=torch.long).unsqueeze(1).unsqueeze(0) * j
        out = model(
            x=batch["x_input"],
            t=t_dense.unsqueeze(1)
            .unsqueeze(0)
            .expand(len(batch["x_input"]), dense_pts, 1)
            .to(device),
            band_idx_part=batch["band_idx"],
            band_idx_full=bands.expand(len(batch["x_input"]), dense_pts, 1).to(
                device
            ),
            mask=batch["pad_mask_part"],  # Needs to be bands for decoding per time!
            time_sorted=batch['time_sorted'],
            sequence_batch_mask=batch['sequence_batch_mask'],
        )

        reconstruction = (
            out["reconstructed"][idx].detach().cpu().numpy()[:, 0]
        )
        mask = band == j
        if not experiment.inverse_scale_reconstruction: # This will make log-transformed decoded fluxes predict the original flux in the true flux space
            scale = experiment.data.dataset.flux_normalizer.max_value
            reconstruction = (reconstruction + 0.5) / scale - 0.5 # we want predicted scaled flux f is flux_norm(semilog(F))

        reconstruction = experiment.data.dataset.inverse_transform(reconstruction)
        

        plt.errorbar(
            time[mask & sample_mask],
            flux[mask & sample_mask],
            yerr=(flux_err[mask & sample_mask]**2)**0.5,
            marker="o",
            label=f"band {IDX2NAME[j]}",
            ls="none",
            color=BAND_COL[j],
            alpha=0.8,
        )

        plt.errorbar(
            time[mask & ~sample_mask],
            flux[mask & ~sample_mask],
            yerr=(flux_err[mask & ~sample_mask]**2)**0.5,
            marker="o",
            ls="none",
            color=BAND_COL[j],
            mfc='none',
            alpha=0.5
        )
        
        plt.plot(t_dense, reconstruction, color=BAND_COL[j])
    plt.gca().set_title(f'Reconstruction of {class_name}')
    plt.legend(ncol=2)
    plt.xlabel("Scaled Time")
    plt.ylabel("Scaled Flux")
    plt.xlim(-abs(time).max() * 1.2, abs(time).max() * 1.2)
    plt.ylim(-0.5, flux.max()*1.5)

    return fig



def plot_reconstruction_from_batch(experiment, batch, idx_in_batch, *,
                                   dense_pts=300, device=None, ax=None, inverse_scale_reconstruction=True):
    """
    Entry point for plotting a reconstruction from a batch.
    Returns the matplotlib Figure.
    """
    sample = {
        "time"     : batch["time_full"][idx_in_batch],
        "flux"     : batch["flux_full"][idx_in_batch],
        "band_idx" : batch["band_idx_full"][idx_in_batch],
        "sigma"    : batch["flux_err_full"][idx_in_batch],
        "x_input"  : batch["x_input"][idx_in_batch]
    }

    return plot_full_multiband_dense_plot(experiment, batch, idx_in_batch, device,
                                     dense_pts=dense_pts, ax=ax, inverse_scale_reconstruction=inverse_scale_reconstruction)