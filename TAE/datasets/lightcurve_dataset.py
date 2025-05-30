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
from enum import Enum
from TAE.util import fill_config, extract_curves_from_filtered_data



# Mapping of band names to integer codes
BAND_IDX_MAP = {"g": 0, "r": 1, "i": 2, "z": 3, "y": 4, "u": 5}

class LightCurvePreprocessor(Dataset):

    """
    Light curve preprocessor that applies augmentation and normalization to light curves.
    """

    def __init__(
        self, 
        full_light_curves,
        flux_normalization="max_partial_scalar", 
        log_flux=True, 
        time_normalization="minmax", 
        augmentation="mixed", 
        min_points=5, 
        min_obs_fraction=0.5,
        num_aug_per_curve=3,
    ):
        self.full_light_curves = full_light_curves
        self.flux_normalization = flux_normalization
        self.log_flux = log_flux
        self.time_normalization = time_normalization
        self.augmentation = augmentation
        self.min_points = min_points
        self.min_obs_fraction = min_obs_fraction

        # Build (light curve index, augmentation mode) list
        self.index_pairs = []
        for i, full_lc in enumerate(self.full_light_curves):
            if len(full_lc) < min_points:
                continue
            self.index_pairs.append((i, "full"))
            for _ in range(num_aug_per_curve):
                if self.augmentation == "mixed":
                    mode = np.random.choice(["cutoff", "cuton", "sparse"])
                else:
                    mode = self.augmentation
                self.index_pairs.append((i, mode))

        columns = ["days_since_first_observation", "FLUXCAL", "FLUXCALERR"]

        self.full_light_curve_array = [
            df[columns].astype(np.float32).reset_index(drop=True).values
            for df in tqdm(self.full_light_curves)
        ]

        self.band_array = [
            df["band" if "band" in df.columns else "filter"]
            .reset_index(drop=True)
            .values
            for df in self.full_light_curves
        ]

        print(
            f"Dataset contains {len(self.index_pairs)} samples from {len(full_light_curves)} full curves."
        )

    def apply_augmentation(self, lc, lc_band, mode):

        n = len(lc) 

        min_len = max(self.min_points, int(self.min_obs_fraction * n))
        if mode == "full" or (n <= min_len):
            return lc, lc_band, np.arange(n)
        if mode == "cutoff":
            aug_idx = np.random.randint(min_len, n)
            return lc[:aug_idx], lc_band[:aug_idx], np.arange(aug_idx)
        if mode == "cuton":
            hi = n - min_len
            aug_idx = np.random.randint(hi, n)
            return lc[aug_idx:], lc_band[aug_idx:], np.arange(aug_idx, n)
        if mode == "sparse":
            k = np.random.randint(min_len, n)
            aug_idx = np.sort(np.random.choice(n, k, replace=False))
            return lc[aug_idx], lc_band[aug_idx], aug_idx
        else:
            raise NameError(f'mode must be one of ("cutoff", "cuton", "sparse", "full") but got {mode}')
        
        

    def normalize_flux(self, flux, fluxerr, flux_norm_info=None):

        if self.log_flux:
            flux = np.sign(flux) * np.log10(np.abs(flux) + 1.0)
            fluxerr = np.sign(flux) * fluxerr / (np.abs(flux) + 1.0) / np.log(10)

        if self.flux_normalization == "max_partial_scalar":
            if flux_norm_info is None: 
                scalar = flux.max() + 1e-6
            else:
                scalar = flux_norm_info[3]
            flux = flux / scalar
            fluxerr = fluxerr / scalar
            fluxnorm_info = ("log_flux", self.log_flux, "max_partial_scalar", float(scalar))

        if self.flux_normalization == "zscore":
            if flux_norm_info is None: 
                mu = flux.mean()
                sd = flux.std() or 1.0
            else:
                mu, sd = flux_norm_info[3], flux_norm_info[4]
            flux = (flux - mu) / sd
            fluxerr = fluxerr / sd
            fluxnorm_info = ("log_flux", self.log_flux, "zscore", float(mu), float(sd))

        return flux, fluxerr, fluxnorm_info

    def normalize_time(self, time, time_norm_info=None):
        if self.time_normalization == "minmax":
            if time_norm_info is None:
                t_min = time.min()
                t_max = 500
            else:
                t_min, t_max = time_norm_info[1], time_norm_info[2]
            time = (time - t_min) / t_max
            return time, ("minmax", float(t_min), float(t_max))
        else:
            return time, ("none")
        
    def __len__(self):

        return len(self.index_pairs)

    def __getitem__(self, idx):
        
        '''
        Get the light curve and preform preprocessing 
        (augmentation, flux/time normalization) on the light curve.
        For augmented light curves, return the full light curve as well.
        '''
        # get the original light curve and augmentation mode
        lc_idx, mode = self.index_pairs[idx]
        lc_full = self.full_light_curve_array[lc_idx].copy()  ## TIME, FLUX, FLUXERR
        band_full = self.band_array[lc_idx].copy()
        band_idx_full = np.array([BAND_IDX_MAP[str(b)] for b in self.band_array[lc_idx].copy()])

        # Apply augmentation        
        lc_part, band_part, aug_idx  = self.apply_augmentation(lc_full, band_full, mode)
        band_idx_part = np.array([BAND_IDX_MAP[str(b)] for b in band_part])

        # Normalize time
        time_part, time_norm_info  = self.normalize_time(lc_part[:, 0])
        time_full, _ = self.normalize_time(lc_full[:, 0], time_norm_info=time_norm_info)

        # Normalize flux
        flux_part, flux_err_part, flux_norm_info =  self.normalize_flux(lc_part[:,1],lc_part[:,2])
        flux_full, flux_err_full, _ =  self.normalize_flux(lc_full[:,1],lc_full[:,2], flux_norm_info=flux_norm_info)

        # Mask for in-sample points
        mask = np.zeros(len(time_full), dtype=bool)
        mask[aug_idx] = True
        scalar_t_part = np.full_like(time_part, flux_norm_info[-1])
        x_input = np.vstack([time_part, flux_part, scalar_t_part]).T

        return {
            "x_input": torch.from_numpy(x_input),
            "time_part": torch.from_numpy(time_part).unsqueeze(1),
            "flux_part": torch.from_numpy(flux_part).unsqueeze(1),
            "flux_err_part": torch.from_numpy(flux_err_part).unsqueeze(1),
            "band_idx_part": torch.from_numpy(band_idx_part).unsqueeze(1),
            "time_full": torch.from_numpy(time_full).unsqueeze(1),
            "flux_full": torch.from_numpy(flux_full).unsqueeze(1),
            "flux_err_full": torch.from_numpy(flux_err_full).unsqueeze(1),
            "band_idx_full": torch.from_numpy(band_idx_full).unsqueeze(1),
            "in_sample_mask": torch.from_numpy(mask).unsqueeze(1),
            "peak_time": torch.tensor(float(time_full.max())),
            "peak_flux": torch.tensor(float(flux_full.max())),
            "time_norm_info": time_norm_info,
            "flux_norm_info": flux_norm_info,
            "augmentation": mode,  
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
            out[i, : t.shape[0]] = t
        return out

    coll = {
        "x_input": pad([b["x_input"] for b in batch], (B, max_part, 3)),
        "band_idx": pad([b["band_idx_part"] for b in batch], (B, max_part, 1), torch.long),
        "time_full": pad([b["time_full"] for b in batch], (B, max_full, 1)),
        "flux_full": pad([b["flux_full"] for b in batch], (B, max_full, 1)),
        "flux_err_full": pad([b["flux_err_full"] for b in batch], (B, max_full, 1)),
        "band_idx_full": pad(
            [b["band_idx_full"] for b in batch], (B, max_full, 1), torch.long
        ),
        "in_sample_mask": pad(
            [b["in_sample_mask"] for b in batch],
            (B, max_full, 1),
            torch.bool,
        ).squeeze(2),
        "lengths": torch.tensor([len(b["x_input"]) for b in batch]),
        "full_lengths": torch.tensor([len(b["time_full"]) for b in batch]),
        "peak_time": torch.tensor([b["peak_time"] for b in batch]),
        "peak_flux": torch.tensor([b["peak_flux"] for b in batch]),
        "flux_norm_info": [b["flux_norm_info"] for b in batch],
        "time_norm_info": [b["time_norm_info"] for b in batch]
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

    def __init__(
        self,
        lightcurve_path=".",
        batch_size=32,
        num_workers=4,
        flux_normalization="max_partial_scalar",
        time_normalization="minmax",
        log_flux=True,
        min_points=5,
        augmentation="mixed",
        num_aug_per_curve=3,
        min_obs_fraction=0.5,
        class_filter=None,
    ):
        super().__init__()
        self.min_obs_fraction = min_obs_fraction
        self.lightcurve_path = lightcurve_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flux_normalization = flux_normalization
        self.time_normalization = time_normalization
        self.log_flux = log_flux
        self.min_points = min_points
        self.augmentation = augmentation
        self.num_aug_per_curve = num_aug_per_curve
        self.dataset = None
        self.class_filter = class_filter

        with open(lightcurve_path, "rb") as f:
            filtered_data = pickle.load(f)

        self.full_light_curves = extract_curves_from_filtered_data(
            filtered_data, min_points=min_points, class_filter=class_filter
        )

    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = LightCurvePreprocessor(
                self.full_light_curves,
                flux_normalization=self.flux_normalization, 
                log_flux=self.log_flux, 
                time_normalization=self.time_normalization, 
                augmentation=self.augmentation,
                num_aug_per_curve=self.num_aug_per_curve,
                min_obs_fraction=self.min_obs_fraction,
            )
        dataset = self.dataset

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            prefetch_factor=10,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            prefetch_factor=10,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            prefetch_factor=10,
        )
