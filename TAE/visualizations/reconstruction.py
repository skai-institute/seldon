import torch
import matplotlib.pyplot as plt
import math

import numpy as np


def linear_to_log(f_lin):
    return np.sign(f_lin) * np.log10(np.absolute(f_lin) + 1.0)

def log_to_linear(f_log):
    return np.sign(f_log) * (10.0 ** np.absolute(f_log) - 1.0)

def sigma_log_to_linear(s_log, f_lin):
    #return s_log * (10.0 ** np.absolute(f_log)) * math.log(10.0)
    return s_log * (np.absolute(f_lin) + 1.0) * np.log(10.0)

#BAND_COL = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
BAND_COL = ["#4daf4a", "#e41a1c", "#ff7f00", "#333333", "#984ea3", "#377eb8"]
IDX2NAME = {0:"g", 1:"r", 2:"i", 3:"z", 4:"y", 5:"u"}
TIME_SCALE = 1.0
BAND_IDX_MAP = {"g":0, "r":1, "i":2, "z":3, "y":4, "u":5}


def plot_full_multiband_dense_plot(model, batch, idx, device, ax=None, dense_pts=300):

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
    if batch['flux_norm_info'][idx][2] == 'minmax':
        flux += 0.5
        flux *= batch['flux_norm_info'][idx][-1]
        flux_err *= batch['flux_norm_info'][idx][-1]
    elif batch['flux_norm_info'][idx][2] == 'max_partial_scalar':
        flux *=  batch['flux_norm_info'][idx][-1]
        flux_err *= batch['flux_norm_info'][idx][-1]
    flux = log_to_linear(flux)
    flux_err = sigma_log_to_linear(flux_err, flux)
    for j in range(6):
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
            mask=batch["in_sample_mask"],  # Needs to be bands for decoding per time!
            time_sorted=batch['time_sorted'],
            sequence_batch_mask=batch['sequence_batch_mask'],
        )

        reconstruction = (
            out["reconstructed"][idx].detach().cpu().numpy()[:, 0]
        )
        mask = band == j
        if batch['flux_norm_info'][idx][2] == 'minmax':
            reconstruction += 0.5
            reconstruction *= batch['flux_norm_info'][idx][-1]

        reconstruction = log_to_linear(reconstruction)
        

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

@torch.no_grad()
def full_multiband_dense_plot(model, sample, *, dense_pts = 300, device = None,):
    """
    Multiband dense reconstruction with error bars and observed points.
    Return the Matplotlib figure object.
    """
    t_all = sample['time'].unsqueeze(0)
    f_all = sample['flux'].unsqueeze(0)   
    b_all = sample["band_idx"].unsqueeze(0)
    z     = model.encoder(t_all, f_all, b_all)
    
    t_min, t_max = sample["time"].min(), sample["time"].max()
    t_dense = torch.linspace(t_min, t_max, dense_pts).view(1,-1,1)
    
    fig, ax = plt.subplots(figsize = (7,3.5))
    
    for code in torch.unique(sample['band_idx']):
        code = int(code)
        colour, name = BAND_COL(code), IDX2NAME(code)
        
        # Dense reconstruction
        b_dense = torch.full_like(t_dense, code, dtype = torch.long)
        f_dense_log = model.decoder(z, t_dense, b_dense)
        t_dense_cpu = (t_dense.squeeze()*TIME_SCALE).cpu()
        f_dense_cpu = log_to_linear(f_dense_log.squeeze()).cpu()
        ax.plot(t_dense_cpu, f_dense_cpu, '-', lw = 2.2, color = colour, label = f"{name} recon")
        
        # Observed points
        mask = (sample["band_idx"].squeeze() == code)
        t_obs = (sample["time"][mask] * TIME_SCALE).cpu()
        f_log = sample["flux"][mask]
        f_obs = log_to_linear(f_log).cpu()
        ax.scatter(t_obs, f_obs, s=28, color=colour, alpha=0.9)

        # Optional error bars
        if "sigma" in sample:
            s_lin = sigma_log_to_linear(sample["sigma"][mask], f_log).abs().cpu()
            ax.errorbar(t_obs, f_obs, yerr=s_lin,
                        fmt='none', ecolor=colour, elinewidth=1,
                        alpha=0.5, capsize=1.5)
    ax.set_xlabel("days")
    ax.set_ylabel("FLUXCAL")
    ax.set_title("Full-curve multiband reconstruction")
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_reconstruction_from_batch(model, batch, idx_in_batch, *,
                                   dense_pts=300, device=None, ax=None):
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

    return plot_full_multiband_dense_plot(model, batch, idx_in_batch, device,
                                     dense_pts=dense_pts, ax=ax)