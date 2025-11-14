# Visualization and Analysis Tools
import matplotlib.pyplot as plt
import numpy as np
import torch


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os


def plot_enhanced_predictions(
    model, dataset, index, device=None, fractions=[0.1, 0.3, 0.7, 1.0]
):
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
    norm_info = sample["norm_info"]
    t_full = sample["time_full"].squeeze().numpy()
    flux_full = sample["flux_full"].squeeze().numpy()
    flux_err_full = sample.get("flux_err_full", None)
    if flux_err_full is not None:
        flux_err_full = flux_err_full.squeeze().numpy()

    true_peak_time = sample["peak_time"].item()
    true_peak_flux = sample["peak_flux"].item()

    cmap = plt.cm.get_cmap("tab10", len(fractions))
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "^", "s", "P"]

    fig, ax = plt.subplots(figsize=(13, 6))

    # === Full light curve
    ax.errorbar(
        t_full,
        flux_full,
        yerr=flux_err_full if flux_err_full is not None else None,
        fmt="x",
        color="gray",
        alpha=0.3,
        label="Full Curve",
    )

    ax.axvline(
        x=true_peak_time,
        color="green",
        linestyle="-",
        label=f"True Peak: {true_peak_time:.1f}d",
    )

    # === Dense time grid for prediction
    t_dense = torch.linspace(min(t_full), max(t_full), 200).unsqueeze(1).to(device)

    predictions = []
    seen_indices = set()

    for i, frac in enumerate(fractions):
        cutoff = max(1, int(frac * len(sample["x_input"])))
        x_partial = sample["x_input"][:cutoff].unsqueeze(0).to(device)

        with torch.no_grad():
            z = model.encode(x_partial)
            full_lengths = torch.full(
                (z.size(0),), t_dense.size(0), dtype=torch.long, device=z.device
            )
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

        ax.plot(
            t_dense_np,
            recon_denorm,
            linestyle=linestyles[i % len(linestyles)],
            color=color,
            label=f"{int(frac * 100)}% Recon (Peak: {pred_peak_time:.1f}d)",
        )
        ax.axvline(
            pred_peak_time,
            color=color,
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1,
        )

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
            ax.errorbar(
                t_part[new_mask],
                f_part[new_mask],
                yerr=ferr_part[new_mask],
                fmt=markers[i % len(markers)],
                capsize=3,
                color=color,
                label=f"{int(frac * 100)}% Partial Obs",
            )
        else:
            ax.scatter(
                t_part[new_mask],
                f_part[new_mask],
                s=50,
                edgecolors="black",
                marker=markers[i % len(markers)],
                color=color,
                label=f"{int(frac * 100)}% Partial Obs",
            )

        predictions.append(
            {
                "fraction": frac,
                "peak_time": pred_peak_time,
                "peak_flux": pred_peak_flux,
                "time_error": abs(pred_peak_time - true_peak_time),
                "flux_error": abs(pred_peak_flux - true_peak_flux)
                / (true_peak_flux + 1e-6)
                * 100,
            }
        )

    # === Summary table
    textstr = (
        f"True Peak: {true_peak_time:.1f}d, {true_peak_flux:.1f} flux\n"
        + "-" * 40
        + "\n"
    )
    textstr += f"{'Frac':>6s}  {'Time':>8s}  {'Error':>8s}  {'Flux Err%':>10s}\n"
    for p in predictions:
        textstr += f"{int(p['fraction'] * 100):3d}%   {p['peak_time']:6.1f}d   {p['time_error']:6.1f}d   {p['flux_error']:8.1f}%\n"

    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(
        0.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=props,
        family="monospace",
    )

    ax.set_title(
        f"Light Curve {index} — Reconstructions and Peak Predictions", fontsize=14
    )
    ax.set_xlabel("Days Since First Observation")
    ax.set_ylabel("FLUXCAL")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    return fig, predictions




# --- metric helpers --------------------
def _fractional_errors(pred, truth, eps=1e-8):
    fe = (pred - truth).abs() / (truth.abs() + eps)
    return fe.mean(1).squeeze(-1), fe.max(1).values.squeeze(-1)

def _select(vec: torch.Tensor, frac: float = 0.10):
    order = torch.argsort(vec)                 # ascending = “good” first
    n     = max(1, int(len(vec)*frac))
    mid0  = len(vec)//2 - n//2
    return {"best":  order[:n].tolist(),
            "median":order[mid0:mid0+n].tolist(),
            "worst": order[-n:].tolist()}

# ---------- colour palette & map -----------------------------------------
BAND_COL = ["#e41a1c", "#377eb8", "#4daf4a",
            "#984ea3", "#ff7f00", "#ffff33"]   #  g r i z y u
IDX2NAME = {v:k for k,v in BAND_IDX_MAP.items()}

# ---------- improved plot helper -----------------------------------------
def _plot_single(t, y_true, y_pred, title="", band_idx=None):
    """
    Parameters
    ----------
    t        : (T,1)  tensor
    y_true   : (T,1)
    y_pred   : (T,1)
    band_idx : (T,1) or None
    """
    t_np   = t.cpu().numpy().squeeze()
    y_t_np = y_true.cpu().numpy().squeeze()
    y_p_np = y_pred.cpu().numpy().squeeze()

    plt.figure()

    if band_idx is None:                        # single-colour fallback
        plt.scatter(t_np, y_t_np, s=12, alpha=.8, label="truth")
    else:
        b_np = band_idx.cpu().numpy().squeeze()
        for code in np.unique(b_np):
            mask = b_np == code
            name = IDX2NAME.get(int(code), str(code))
            plt.scatter(t_np[mask], y_t_np[mask],
                        s=14, alpha=.8,
                        color=BAND_COL[int(code)],
                        label=f"{name}")

    plt.plot(t_np, y_p_np, "-", lw=1.7, color="orange", label="recon")
    plt.xlabel("time"); plt.ylabel("flux")
    plt.title(title); plt.legend(); plt.show()

# ------------------------------ NEW evaluate_batch ------------------------------
def evaluate_batch(model, batch, *, device=None, frac=0.10,
                   nbins=50, make_plots=True, eps_den=1e-8):
    """
    Evaluate one collated batch in **linear FLUXCAL units**.

    Returns
    -------
    {
      'metrics': {'MeanFE': Tensor(B), 'MaxFE': Tensor(B),
                  'mean|Z|': Tensor(B), 'max|Z|': Tensor(B)},
      'indices': {metric: {'best': [...], 'median': [...], 'worst': [...]}}
    }
    """
    dev   = device or next(model.parameters()).device
    batch = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in batch.items()}

    with torch.no_grad():
        pred_compressed = model(batch)             # (B,T,1)  arcsinh space

    # ---------- back to linear flux --------------------------------------
    true_lin = torch.sinh(batch["flux"])          * LOG_EPS        # (B,T,1)
    pred_lin = torch.sinh(pred_compressed)        * LOG_EPS
    sigma_lin= batch["sigma"]                     * LOG_EPS

    # ---------- fractional errors ----------------------------------------
    fe_mean, fe_max = _fractional_errors(pred_lin, true_lin, eps=eps_den)

    # ---------- |Z|-scores ------------------------------------------------
    z_abs  = (pred_lin - true_lin).abs() / (sigma_lin + eps_den)
    z_mean = z_abs.mean(1).squeeze(-1)
    z_max  = z_abs.max (1).values.squeeze(-1)

    metrics = {"MeanFE": fe_mean,
               "MaxFE" : fe_max,
               "mean|Z|": z_mean,
               "max|Z|" : z_max}
    indices = {k: _select(v, frac) for k,v in metrics.items()}

    # ---------- optional histograms & examples ---------------------------
    if make_plots:
        for name, vec in metrics.items():
            plt.figure()
            plt.hist(vec.cpu(), bins=nbins, density=True, alpha=.8)
            plt.xlabel(name); plt.ylabel("PDF"); plt.title(name)
            plt.show()

        tag_metric = "MeanFE"
        # ––– inside evaluate_batch  ---------------------------------------------
        for idx in (indices[tag_metric]["best"][:1] +
                    indices[tag_metric]["worst"][:1]):
            _plot_single(batch["time"][idx],
                        true_lin[idx],
                        pred_lin[idx],
                        title=f"idx {idx} ({tag_metric})",
                        band_idx=batch["band_idx"][idx])   #  ← add this line


    return {"metrics": {k: v.cpu() for k,v in metrics.items()},
            "indices": indices}
