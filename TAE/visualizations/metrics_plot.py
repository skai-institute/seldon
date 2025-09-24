import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from TAE.util import log_to_linear, sigma_log_to_linear

BAND_IDX_MAP = {"g":0, "r":1, "i":2, "z":3, "y":4, "u":5}

def plot_metric_histograms(metrics_dict, n_bins = 50):
    """
    Plot histograms for all metrics in a dictionary.
    Returns a list of matplotlib Figure objects.
    """
    figs = []
    #results = MetricsEvaluator(model, batch).all_metrics()
    for name, vec in metrics_dict.items():
        fig, ax = plt.subplots()
        ax.hist(vec.cpu(), bins = n_bins, density = True, alpha = 0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("PDF")
        ax.set_title(name)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_metric_corner(metrics_dict):
    """
    Make a Seaborn corner plot of all metrics.
    Returns the matplotlib Figure.
    """
    df = pd.DataFrame({k:v.cpu().numpy() for k,v in metrics_dict.items()})
    g = sns.pairplot(df, corner = True, diag_kind = 'hist', plot_kws=dict(s=6, alpha=.3))
    g.fig.suptitle("Metric corner plot", y=1.02)
    g.fig.tight_layout()
    return g.fig
    


def plot_learning_curve(model, batch, *,
                        fractions=np.linspace(0.1, 1.0, 10),
                        metric="MeanFE",
                        device=None):
    """
    Plot Mean Fractional Error vs. fraction of curve revealed.
    """
    dev = device or next(model.parameters()).device
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
    B, L, _ = batch["time_full"].shape
    base_mask = batch["in_sample_mask"]

    vals = []

    for f in fractions:
        # 1) Partial masking
        m_f = torch.zeros_like(base_mask, dtype=torch.bool)
        for i in range(B):
            n_i = base_mask[i].sum().item()
            keep = int(np.ceil(f * n_i))
            idx = base_mask[i].nonzero(as_tuple=False)[:keep]
            m_f[i, idx[:, 0], 0] = True

        # 2) Create partial batch
        batch_part = {k: (v.clone() if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
        batch_part["flux"] = batch_part["flux_full"] * m_f
        batch_part["sigma"] = batch_part["flux_err_full"] * m_f + (~m_f) * 1e6
        batch_part["mask"] = m_f

        with torch.no_grad():
            pred_log = model(batch_part)

        true_lin = log_to_linear(batch_part["flux_full"]+0.5)
        pred_lin = log_to_linear(pred_log+0.5)

        if metric == "MeanFE":
            denom = true_lin.abs().clamp_min(0.1)
            fe = (pred_lin - true_lin).abs() / denom
            mean_fe = (fe * m_f).sum(1) / m_f.sum(1).clamp_min(1)
            vals.append(mean_fe.squeeze(-1).cpu())
        else:
            raise ValueError("Only MeanFE supported in learning curve for now.")

    # Plotting
    val_np = torch.stack(vals, dim=1).cpu().numpy()
    med = np.median(val_np, axis=0)
    p10 = np.quantile(val_np, 0.10, axis=0)
    p90 = np.quantile(val_np, 0.90, axis=0)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(fractions, p10, p90, alpha=0.25, color="C0", label="10–90%")
    ax.plot(fractions, med, "o-", lw=2, color="C0", label="median")
    ax.set_xlabel("Fraction of curve revealed")
    ax.set_ylabel("Mean Fractional Error")
    ax.set_title("Learning curve on partial observations")
    ax.grid(ls=":")
    ax.legend()
    fig.tight_layout()
    return fig