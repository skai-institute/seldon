#!/usr/bin/env python
"""
compare_best_median_worst.py
--------------------------------
Select K best / median / worst light curves under a Logistic-CDF (sigmoid) model
and plot them side-by-side against a Gumbel-CDF model.
"""
import argparse, heapq, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from TAE.util                     import fill_config
from TAE.datasets.dataset_factory import get_module as get_dataset
from TAE.experiments.factory      import get_module as get_experiment
from TAE.models.factory           import get_module
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch


# -----------------------------------------------------------------
def load_exp(logdir, version):
    """Load a Lightning experiment and its validation DataLoader."""
    logdir = Path(logdir)
    cfg = OmegaConf.load(logdir / f"version_{version}" / "hparams.yaml")
    cfg["dataset"]["config"]["num_workers"] = 1
    cfg = fill_config(cfg); conf = OmegaConf.to_container(cfg, resolve=True)

    model = get_module(cfg["model_params"]["name"], conf["model_params"]["config"])
    data  = get_dataset(conf["dataset"]["name"], conf["dataset"]["config"])
    xp    = get_experiment(cfg["exp_params"]["name"],
                           {"model": model, "data": data,
                            **conf["exp_params"]["config"]})

    ckpt = sorted((logdir / f"version_{version}" / "checkpoints").glob("epoch*"))[-1]
    xp   = xp.load_from_checkpoint(ckpt, map_location="cpu",
                                   model=model, data=data,
                                   params=cfg["exp_params"]["config"]["params"])
    xp.eval(); data.setup("validate")
    return xp, data.val_dataloader()


def as_list(x):
    """Return a Python list whether x is tensor or already list."""
    return x.tolist() if isinstance(x, torch.Tensor) else x


# -----------------------------------------------------------------
def main(args):
    log_xp, val_loader = load_exp(args.logdir, args.version_log)
    gum_xp, _          = load_exp(args.logdir, args.version_gum)

    K = args.k
    best_heap, worst_heap, medbuf = [], [], []

    # ---------- Pass 1: compute χ² and keep top-K/bottom-K -------------
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if args.max_batches and b_idx >= args.max_batches:
                break

            res  = log_xp.validation_step(batch, 0)
            chi2 = res["chi2"].cpu().numpy()[:, 0]
            ids  = as_list(batch["raw_idx"])

            for e, lc in zip(chi2, ids):
                heapq.heappush(best_heap,  (-e, lc))
                heapq.heappush(worst_heap, ( e, lc))
                if len(best_heap)  > K: heapq.heappop(best_heap)
                if len(worst_heap) > K: heapq.heappop(worst_heap)
                medbuf.append((e, lc))

    # ---------- derive median IDs --------------------------------------
    medbuf.sort(key=lambda x: x[0])
    mid = len(medbuf) // 2
    median_ids = [lc for _, lc in medbuf[mid - K//2 : mid + K//2]]

    best_ids  = [lc for _, lc in sorted(best_heap,  key=lambda x: -x[0])]
    worst_ids = [lc for _, lc in sorted(worst_heap, key=lambda x:  x[0])]

    #groups = [("BEST", best_ids), ("MEDIAN", median_ids), ("WORST", worst_ids)]
    groups = [("WORST", worst_ids[:K])]      # only K worst curves
    print({tag: ids for tag, ids in groups})

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Pass 2: plot selected curves ---------------------------
    wanted = {lc for _, lst in groups for lc in lst}

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            ids = as_list(batch["raw_idx"])
            for idx_in_batch, lc in enumerate(ids):
                if lc not in wanted:
                    continue

                fig, ax = plt.subplots(1, 2, figsize=(9, 3),
                                       sharex=True, sharey=True)
                plot_reconstruction_from_batch(log_xp.model,  batch,
                                               idx_in_batch, ax=ax[0], device="cpu")
                plot_reconstruction_from_batch(gum_xp.model, batch,
                                               idx_in_batch, ax=ax[1], device="cpu")
                ax[0].set_title(f"Logistic – {lc}")
                ax[1].set_title(f"Gumbel   – {lc}")
                fig.tight_layout()
                tag = next(g for g, lst in groups if lc in lst)
                fig.savefig(outdir / f"{tag}_{lc}.png", dpi=250)
                plt.close(fig)
                wanted.remove(lc)

            if not wanted or (args.max_batches and b_idx >= args.max_batches - 1):
                break


# -----------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True,
                   help="Parent log directory containing version_X folders")
    p.add_argument("--version_log",  type=int, required=True,
                   help="Version number for Logistic/Sigmoid run")
    p.add_argument("--version_gum",  type=int, required=True,
                   help="Version number for Gumbel run")
    p.add_argument("-k", "--k", type=int, default=5,
                   help="Number of curves per bucket (best/median/worst)")
    p.add_argument("--outdir", default="figs_compare",
                   help="Directory to save PNG figures")
    p.add_argument("--max_batches", type=int, default=None,
                   help="Debug: only scan this many batches")
    main(p.parse_args())
