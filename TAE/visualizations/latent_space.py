# visualization/latent_space.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader

from TAE.util import log_to_linear
import umap
BAND_IDX_MAP = {"g":0, "r":1, "i":2, "z":3, "y":4, "u":5}

def per_curve_mse(batch, pred_log):
    true_lin = log_to_linear(batch["flux_full"]) * batch["in_sample_mask"]
    pred_lin = log_to_linear(pred_log) * batch["in_sample_mask"]
    n = batch["in_sample_mask"].sum(1).clamp_min(1)
    return (((pred_lin - true_lin) ** 2).sum(1) / n).squeeze(-1)


def plot_latents(output, 
                method="pca", 
                n_components=2,
                color_by="mse",
                loss_scale="log10",
                label_mode="top_k",     # or "color_threshold"
                color_thr=0.55,
                top_k=20,
                ):
    """
    Perform dimensionality reduction on latent vectors and plot colored by per-curve loss.
    """
    if isinstance(color_by, str):
        assert color_by in {"mse", "class_weight"}
    assert method in {"pca", "umap"}
    assert loss_scale in {"linear", "log", "log10", "sqrt"}

    out = output['out']
    z = out['z_mean'].detach().cpu().numpy()
    loss = output['chi2'].detach().cpu().numpy()

    if color_by == "mse":
        if loss_scale == "linear":
            loss_np = loss
        elif loss_scale == "log":
            loss_np = np.log(loss + 1e-6)
        elif loss_scale == "log10":
            loss_np = np.log10(loss + 1e-6)
        elif loss_scale == "sqrt":
            loss_np = np.sqrt(loss+ 1e-6)
    else:
        loss_np = output['batch'][color_by].detach().cpu().numpy()
        if loss_scale == 'log10':
            loss_np = np.log10(loss_np)

    # Dimensionality reduction
    if method == "pca":
        Z_proj = PCA(n_components=n_components, random_state=42, whiten=True).fit_transform(z[:, 2:])
    else:
        Z_proj = umap.UMAP(n_components=n_components, random_state=42).fit_transform(z)


    # Plotting
    norm = Normalize(vmin=loss_np.min(), vmax=loss_np.max())
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(Z_proj[:, 0], Z_proj[:, 1],
                    c=loss_np, cmap="viridis_r", norm=norm,
                    s=8, alpha=.8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label=f"Per-curve {color_by} ({loss_scale})")

    # Label selection
    if label_mode == "color_threshold":
        mask = norm(loss_np) > color_thr
    else:  # "top_k"
        k = min(top_k, len(loss_np))
        mask = np.zeros_like(loss_np, dtype=bool)
        mask[np.argsort(loss_np)[-k:]] = True

    # Label those points
    #ids_arr = np.asarray(ids)
    #for x, y, i in zip(Z_proj[mask, 0], Z_proj[mask, 1], ids_arr[mask]):
    #    ax.text(x, y, str(i), fontsize=7, ha="center", va="center",
    #            color="k", bbox=dict(facecolor="white", alpha=.7, lw=0, pad=0.4))

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"{method.upper()} on Latent Space Colored by Loss")
    fig.tight_layout()

    return {
        "Z": z,
        "Z_proj": Z_proj,
        "loss": loss_np,
        "fig": fig
    }


@torch.no_grad()
def visualize_latents(model,
                      dataset,
                      collate_fn,
                      *,
                      method="pca",
                      n_components=2,
                      color_by="mse",
                      loss_scale="log10",
                      label_mode="top_k",     # or "color_threshold"
                      color_thr=0.55,
                      top_k=20,
                      batch_size=4096,
                      num_workers=0,
                      shuffle=False,
                      device=None):
    """
    Perform dimensionality reduction on latent vectors and plot colored by per-curve loss.
    """
    assert method in {"pca", "umap"}
    assert color_by in {"mse"}
    assert loss_scale in {"linear", "log", "log10", "sqrt"}

    dev = device or next(model.parameters()).device
    model.eval()

    zs, losses, ids = [], [], []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate_fn, num_workers=num_workers)

    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z = model.encoder(batch["time_part"], batch["flux_part"], batch["band_idx_part"])
        pred = model(batch)
        mse = per_curve_mse(batch, pred)

        zs.append(z.cpu())
        losses.append(mse.cpu())
        ids.extend(batch["idx"])

    Z_lat = torch.cat(zs).numpy()
    loss_t = torch.cat(losses)

    # Apply loss scaling
    if loss_scale == "linear":
        loss_np = loss_t.numpy()
    elif loss_scale == "log":
        loss_np = torch.log(loss_t + 1e-6).numpy()
    elif loss_scale == "log10":
        loss_np = torch.log10(loss_t + 1e-6).numpy()
    elif loss_scale == "sqrt":
        loss_np = torch.sqrt(loss_t + 1e-6).numpy()

    # Dimensionality reduction
    if method == "pca":
        Z_proj = PCA(n_components=n_components, random_state=42).fit_transform(Z_lat)
    else:
        from umap import UMAP
        Z_proj = UMAP(n_components=n_components, random_state=42).fit_transform(Z_lat)

    # Plotting
    norm = Normalize(vmin=loss_np.min(), vmax=loss_np.max())
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(Z_proj[:, 0], Z_proj[:, 1],
                    c=loss_np, cmap="viridis_r", norm=norm,
                    s=8, alpha=.8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label=f"Per-curve MSE ({loss_scale})")

    # Label selection
    if label_mode == "color_threshold":
        mask = norm(loss_np) > color_thr
    else:  # "top_k"
        k = min(top_k, len(loss_np))
        mask = np.zeros_like(loss_np, dtype=bool)
        mask[np.argsort(loss_np)[-k:]] = True

    # Label those points
    ids_arr = np.asarray(ids)
    for x, y, i in zip(Z_proj[mask, 0], Z_proj[mask, 1], ids_arr[mask]):
        ax.text(x, y, str(i), fontsize=7, ha="center", va="center",
                color="k", bbox=dict(facecolor="white", alpha=.7, lw=0, pad=0.4))

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"{method.upper()} on Latent Space Colored by Loss")
    fig.tight_layout()

    return {
        "Z": Z_lat,
        "Z_proj": Z_proj,
        "loss": loss_np,
        "ids": ids_arr,
        "fig": fig
    }
