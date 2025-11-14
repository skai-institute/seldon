# ===== headless CPU env =====
import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless-safe matplotlib
os.environ["OMP_NUM_THREADS"] = "32"

# ===== imports =====
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from TAE.util.loaders import load_model
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch

# -------- user inputs --------
RUN_DIR = "/u/jiezhong/checkpoints_ICLR/BandAwareDeepSetGRUODEVAE/version_2"
OUTDIR  = Path("TAE/figures_more_than_50_SELDON_tgrid1")
OUTDIR.mkdir(parents=True, exist_ok=True)


OVERRIDES = {
    # "dataset.config.lightcurve_path": "/projects/ncsa/caps/skai/data/cleaned_data_8_previous_nondetects_zeroed_merged.pkl",
    # "dataset.config.class_filter": ["SNIa"],
    # "dataset.config.num_workers": 1,
}

experiment = load_model(RUN_DIR, overrides=OVERRIDES) 

# ===== plotting: curves with > 50 points =====
val_loader = experiment.data.val_dataloader()
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        lengths = batch.get("full_lengths", batch["lengths"]).cpu()
        for j in range(len(lengths)):
            if int(lengths[j]) > 50:
                fig = plot_reconstruction_from_batch(
                    experiment.model, batch, idx_in_batch=j, device="cuda"
                )
                out = OUTDIR / f"realtime_prediction_{i}_{j}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out}")
