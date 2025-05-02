import pickle
import numpy as np
import torch
from collections import defaultdict
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch



"""
SNLightCurveDataset module: LightningDataModule wrapper for supernova light curve dataset
Supports optional truncation, z-score or max-so-far normalization, and multiband modeling.
"""

"""
augmented_lightning_dataset.py

LightningDataModule that wraps a custom Dataset for supernova light curves.
Supports band-aware augmentation, normalization, and partial sampling.
"""

# ======================================
# Supernova Light Curve Dataset (Simple Version)
# ======================================


# Mapping of band names to integer codes
BAND_IDX_MAP = {"g": 0, "r": 1, "i": 2, "z": 3, "y": 4, "u": 5}

class SNLightCurveDataset(Dataset):
    """
    Dataset for supernova light curves (single-band).
    Provides [time, normalized flux, scalar] inputs and band code.
    """

    def __init__(self,
                 full_light_curves,
                 normalization="max_partial_scalar",
                 min_points=5,
                 augmentation="mixed",
                 num_aug_per_curve=3,
                 min_obs_fraction=0.5):
        self.full_light_curves = full_light_curves
        self.normalization = normalization
        self.min_points = min_points
        self.augmentation = augmentation
        self.num_aug_per_curve = num_aug_per_curve
        self.min_obs_fraction = min_obs_fraction

        # Build (index, mode) list
        self.index_pairs = []
        for i, full_lc in enumerate(full_light_curves):
            if len(full_lc) < min_points:
                continue
            self.index_pairs.append((i, "full"))
            for _ in range(num_aug_per_curve):
                self.index_pairs.append((i, "aug"))

        print(f"Dataset contains {len(self.index_pairs)} samples from {len(full_light_curves)} full curves.")

    def set_min_obs_fraction(self, frac: float):
        """Optional curriculum helper to adjust minimum observed fraction."""
        self.min_obs_fraction = frac

    def __len__(self):
        return len(self.index_pairs)

    def _apply_augmentation_mode(self, df, mode):
        """Cutoff, cuton, sparse, or return full."""
        n = len(df)
        min_len = max(self.min_points, int(self.min_obs_fraction * n))

        if mode == "cutoff":
            return df.iloc[:np.random.randint(min_len, n)]
        if mode == "cuton":
            hi = n - min_len
            return df.iloc[np.random.randint(0, hi):] if hi > 0 else df
        if mode == "sparse":
            k = np.random.randint(min_len, n)
            return df.iloc[np.sort(np.random.choice(n, k, replace=False))]
        return df

    def _augment(self, df):
        if self.augmentation == "mixed":
            mode = np.random.choice(["cutoff", "cuton", "sparse"])
            return self._apply_augmentation_mode(df, mode)
        elif self.augmentation is not None:
            return self._apply_augmentation_mode(df, self.augmentation)
        else:
            return df

    def __getitem__(self, idx):
        curve_i, mode = self.index_pairs[idx]
        full_df = self.full_light_curves[curve_i].reset_index(drop=True)
        part_df = full_df if mode == "full" else self._augment(full_df)

        # Band code
        band_col = "band" if "band" in full_df.columns else "filter"
        band_code = BAND_IDX_MAP[str(full_df[band_col].iloc[0])]
        band_t = torch.tensor(band_code, dtype=torch.long)

        # Raw arrays
        t_full_raw = full_df["days_since_first_observation"].to_numpy("float32")
        f_full_raw = full_df["FLUXCAL"].to_numpy("float32")
        ferr_full_r = full_df["FLUXCALERR"].to_numpy("float32")

        t_part_raw = part_df["days_since_first_observation"].to_numpy("float32")
        f_part_raw = part_df["FLUXCAL"].to_numpy("float32")
        ferr_part_r = part_df["FLUXCALERR"].to_numpy("float32")

        # Time normalization
        t_min = float(t_full_raw.min())
        t_rng = float(np.ptp(t_full_raw)) + 1e-6
        t_full = (t_full_raw - t_min) / t_rng
        t_part = (t_part_raw - t_min) / t_rng

        # Flux normalization
        if self.normalization == "max_partial_scalar":
            scalar = np.max(f_part_raw) + 1e-6
            f_full = f_full_raw / scalar
            f_part = f_part_raw / scalar
            ferr_full = ferr_full_r / scalar
            ferr_part = ferr_part_r / scalar
            flux_norm = ("max_partial_scalar", float(scalar))
        else:  # "zscore"
            mu, sd = f_full_raw.mean(), f_full_raw.std() or 1.0
            scalar = sd
            f_full = (f_full_raw - mu) / sd
            f_part = (f_part_raw - mu) / sd
            ferr_full = ferr_full_r / sd
            ferr_part = ferr_part_r / sd
            flux_norm = ("zscore", float(mu), float(sd))

        # Convert to torch tensors
        t_full_t = torch.from_numpy(t_full).unsqueeze(1)
        f_full_t = torch.from_numpy(f_full).unsqueeze(1)
        ferr_full = torch.from_numpy(ferr_full).unsqueeze(1)

        t_part_t = torch.from_numpy(t_part).unsqueeze(1)
        f_part_t = torch.from_numpy(f_part).unsqueeze(1)
        ferr_part = torch.from_numpy(ferr_part).unsqueeze(1)

        scalar_t_part = torch.full_like(t_part_t, scalar)
        scalar_t_full = torch.full_like(t_full_t, scalar)

        band_idx_full = band_t.expand(len(t_full_t), 1)
        band_idx_part = band_t.expand(len(t_part_t), 1)

        # Mask for in-sample points
        mask = torch.zeros(len(t_full_t), dtype=torch.bool)
        for tv in t_part_t.squeeze():
            mask |= (torch.abs(t_full_t.squeeze() - tv) < 1e-4)

        # x_input = [time, normalized flux, scalar]
        x_input = torch.cat([t_part_t, f_part_t, scalar_t_part], dim=1)

        return {
            "x_input": x_input,                     # (T_part, 3)
            "band_idx": band_idx_part,               # (T_part, 1)
            "time_partial": t_part_t,
            "flux_partial": f_part_t,
            "flux_err_partial": ferr_part,

            "time_full": t_full_t,
            "flux_full": f_full_t,
            "flux_err_full": ferr_full,
            "scalar_full": scalar_t_full,
            "band_idx_full": band_idx_full,          # (T_full, 1)

            "in_sample_mask": mask,
            "full_lengths": torch.tensor(len(t_full), dtype=torch.long),

            "peak_time": torch.tensor(float(full_df["days_since_first_observation"][f_full_raw.argmax()])),
            "peak_flux": torch.tensor(float(f_full_raw.max())),

            "norm_info": {"flux": flux_norm, "time": ("minmax", float(t_min), float(t_rng))}
        }

# ======================================
# Collate Function for Dataloader
# ======================================

def enhanced_collate_fn(batch):
    """
    Pads a batch of light curves (partial and full) for training.
    """
    B = len(batch)
    max_part = max(len(b["x_input"]) for b in batch)
    max_full = max(len(b["time_full"]) for b in batch)

    def pad(tensors, shape, dtype=None):
        out = torch.zeros(shape, dtype=dtype or tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    coll = {
        "x_input": pad([b["x_input"] for b in batch], (B, max_part, 3)),
        "band_idx": pad([b["band_idx"] for b in batch], (B, max_part, 1), torch.long),

        "time_full": pad([b["time_full"] for b in batch], (B, max_full, 1)),
        "flux_full": pad([b["flux_full"] for b in batch], (B, max_full, 1)),
        "band_idx_full": pad([b["band_idx_full"] for b in batch], (B, max_full, 1), torch.long),
        "in_sample_mask": pad([b["in_sample_mask"].unsqueeze(1) for b in batch],
                              (B, max_full, 1), torch.bool).squeeze(2),

        "lengths": torch.tensor([len(b["x_input"]) for b in batch]),
        "full_lengths": torch.tensor([len(b["time_full"]) for b in batch]),

        "peak_time": torch.tensor([b["peak_time"] for b in batch]),
        "peak_flux": torch.tensor([b["peak_flux"] for b in batch]),
        "norm_info": [b["norm_info"] for b in batch],
    }
    return coll

# ======================================
# PyTorch Lightning DataModule
# ======================================

class SNDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that wraps the SNLightCurveDataset.
    Handles train/val/test splits and DataLoader creation.
    """

    def __init__(self, full_light_curves, batch_size=32, num_workers=4,
                 normalization="max_partial_scalar", min_points=5,
                 augmentation="mixed", num_aug_per_curve=3, min_obs_fraction=0.5):
        super().__init__()
        self.min_obs_fraction = min_obs_fraction
        self.full_light_curves = full_light_curves
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization = normalization
        self.min_points = min_points
        self.augmentation = augmentation
        self.num_aug_per_curve = num_aug_per_curve

    def setup(self, stage=None):
        dataset = SNLightCurveDataset(
            self.full_light_curves,
            normalization=self.normalization,
            min_points=self.min_points,
            augmentation=self.augmentation,
            num_aug_per_curve=self.num_aug_per_curve,
            min_obs_fraction=self.min_obs_fraction
        )


        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=enhanced_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=enhanced_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=enhanced_collate_fn)
