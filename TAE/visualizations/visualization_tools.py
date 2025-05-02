# Visualization and Analysis Tools
import matplotlib.pyplot as plt
import numpy as np
import torch


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os


def plot_enhanced_predictions(model, dataset, index, device=None, fractions=[0.1, 0.3, 0.7, 1.0]):
    """
    Visualizes reconstruction and peak prediction for a single light curve.

    Args:
        model: trained BandAwareAutoencoder
        dataset: SNLightCurveDataset
        index: sample index
        device: torch device
        fractions: list of observed fractions for partial reconstructions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # === Load sample ===
    sample = dataset[index]
    norm_info = sample['norm_info']
    t_full = sample['time_full'].squeeze().numpy()
    flux_full = sample['flux_full'].squeeze().numpy()
    flux_err_full = sample.get("flux_err_full", None)
    if flux_err_full is not None:
        flux_err_full = flux_err_full.squeeze().numpy()

    true_peak_time = sample['peak_time'].item()
    true_peak_flux = sample['peak_flux'].item()

    cmap = plt.cm.get_cmap("tab10", len(fractions))
    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', '^', 's', 'P']

    fig, ax = plt.subplots(figsize=(13, 6))

    # === Full light curve
    ax.errorbar(
        t_full, flux_full, yerr=flux_err_full if flux_err_full is not None else None,
        fmt='x', color='gray', alpha=0.3, label="Full Curve"
    )

    ax.axvline(x=true_peak_time, color='green', linestyle='-', label=f"True Peak: {true_peak_time:.1f}d")

    # === Dense time grid for prediction
    t_dense = torch.linspace(min(t_full), max(t_full), 200).unsqueeze(1).to(device)

    predictions = []
    seen_indices = set()

    for i, frac in enumerate(fractions):
        cutoff = max(1, int(frac * len(sample["x_input"])))
        x_partial = sample["x_input"][:cutoff].unsqueeze(0).to(device)

        with torch.no_grad():
            z = model.encode(x_partial)
            full_lengths = torch.full((z.size(0),), t_dense.size(0), dtype=torch.long, device=z.device)
            recon = model.decode(z, t_dense.unsqueeze(0), full_lengths)

        recon = recon[0].squeeze().cpu().numpy()
        t_dense_np = t_dense.squeeze().cpu().numpy()

        # === Denormalize
        if norm_info[0] == "zscore":
            mu, std = norm_info[1], norm_info[2]
            recon_denorm = recon * std + mu
        elif norm_info[0] == "max_partial_scalar":
            scalar = norm_info[1]
            recon_denorm = recon * scalar
        else:
            recon_denorm = recon

        # === Find predicted peak
        peak_idx = np.argmax(recon_denorm)
        pred_peak_time = t_dense_np[peak_idx]
        pred_peak_flux = recon_denorm[peak_idx]

        color = cmap(i)

        ax.plot(t_dense_np, recon_denorm, linestyle=linestyles[i % len(linestyles)],
                color=color, label=f"{int(frac * 100)}% Recon (Peak: {pred_peak_time:.1f}d)")
        ax.axvline(pred_peak_time, color=color, linestyle=linestyles[i % len(linestyles)], linewidth=1)

        # === New points only
        t_part = sample["time_partial"][:cutoff].squeeze().numpy()
        f_part = sample["flux_partial"][:cutoff].squeeze().numpy()
        ferr_part = sample.get("flux_err_partial", None)
        if ferr_part is not None:
            ferr_part = ferr_part[:cutoff].squeeze().numpy()

        if norm_info[0] == "zscore":
            f_part = f_part * std + mu
            if ferr_part is not None:
                ferr_part = ferr_part * std
        elif norm_info[0] == "max_partial_scalar":
            f_part = f_part * scalar
            if ferr_part is not None:
                ferr_part = ferr_part * scalar

        new_mask = np.array([j not in seen_indices for j in range(cutoff)])
        seen_indices.update(range(cutoff))

        if ferr_part is not None:
            ax.errorbar(t_part[new_mask], f_part[new_mask], yerr=ferr_part[new_mask],
                        fmt=markers[i % len(markers)], capsize=3, color=color,
                        label=f"{int(frac * 100)}% Partial Obs")
        else:
            ax.scatter(t_part[new_mask], f_part[new_mask], s=50, edgecolors='black',
                       marker=markers[i % len(markers)], color=color,
                       label=f"{int(frac * 100)}% Partial Obs")

        predictions.append({
            "fraction": frac,
            "peak_time": pred_peak_time,
            "peak_flux": pred_peak_flux,
            "time_error": abs(pred_peak_time - true_peak_time),
            "flux_error": abs(pred_peak_flux - true_peak_flux) / (true_peak_flux + 1e-6) * 100
        })

    # === Summary table
    textstr = f"True Peak: {true_peak_time:.1f}d, {true_peak_flux:.1f} flux\n" + "-" * 40 + "\n"
    textstr += f"{'Frac':>6s}  {'Time':>8s}  {'Error':>8s}  {'Flux Err%':>10s}\n"
    for p in predictions:
        textstr += f"{int(p['fraction'] * 100):3d}%   {p['peak_time']:6.1f}d   {p['time_error']:6.1f}d   {p['flux_error']:8.1f}%\n"

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')

    ax.set_title(f"Light Curve {index} — Reconstructions and Peak Predictions", fontsize=14)
    ax.set_xlabel("Days Since First Observation")
    ax.set_ylabel("FLUXCAL")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()

    return fig, predictions

