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
        full_class_names,
        flux_normalization="max_partial_scalar",
        log_flux=True,
        time_normalization="minmax",
        augmentation="mixed",
        min_points=5,
        min_obs_fraction=0.5,
        num_aug_per_curve=3,
        t_max=200,
    ):
        self.full_light_curves = full_light_curves
        self.full_class_names = full_class_names
        self.flux_normalization = flux_normalization
        self.log_flux = log_flux
        self.time_normalization = time_normalization
        self.augmentation = augmentation
        self.min_points = min_points
        self.min_obs_fraction = min_obs_fraction
        self.t_max = t_max

        # Build (light curve index, augmentation mode) list
        self.index_pairs = []
        for i, full_lc in enumerate(self.full_light_curves):
            if len(full_lc) < min_points:
                continue
            self.index_pairs.append((i, self.augmentation))
            #for _ in range(num_aug_per_curve):
            #    if self.augmentation == "mixed":
            #        mode = np.random.choice(["cutoff", "cuton", "sparse", "full"])
            #    else:
            #        mode = self.augmentation
            #    self.index_pairs.append((i, mode))

        columns = ["days_since_first_observation", "FLUXCAL", "FLUXCALERR"]

        self.full_light_curve_array = [
            df[columns].astype(np.float32).reset_index(drop=True).values
            for df in tqdm(self.full_light_curves, desc="Extracting LCs")
        ]

        self.band_array = [
            df["band" if "band" in df.columns else "filter"]
            .map(BAND_IDX_MAP)
            .reset_index(drop=True)
            .values
            for df in tqdm(self.full_light_curves, desc="Mapping Bands to Indices")
        ]

        self.full_lengths = [len(arr) for arr in tqdm(self.band_array, desc="Getting Lengths")]
        self.max_length = max(self.full_lengths)

        if self.t_max == "infer":
            ts = np.array([lc[:, 0].max()-lc[:, 0].min() for lc in self.full_light_curve_array])
            self.t_max = 2*ts.std() # +/- 1 sigma
            print("Inferred t_max =", self.t_max)
        print("Found Maximum Light Curve Length:", self.max_length)

        print(
            f"Dataset contains {len(self.index_pairs)} samples from {len(full_light_curves)} full curves."
        )

    def apply_augmentation(self, lc, lc_band, mode):

        n = len(lc)
        if self.augmentation == "mixed":
            mode = np.random.choice(["cutoff", "cuton", "sparse", "full"])
        else:
            mode = self.augmentation

        min_len = max(self.min_points, int(self.min_obs_fraction * n))
        if mode == "full" or (n <= min_len):
            return lc, lc_band, np.arange(n), mode
        if mode == "cutoff":
            aug_idx = np.random.randint(min_len, n)
            return lc[:aug_idx], lc_band[:aug_idx], np.arange(aug_idx), mode
        if mode == "cuton":
            hi = n - min_len
            aug_idx = np.random.randint(0, hi)
            return lc[aug_idx:], lc_band[aug_idx:], np.arange(aug_idx, n), mode
        if mode == "sparse":
            k = np.random.randint(min_len, n)
            aug_idx = np.sort(np.random.choice(n, k, replace=False))
            return lc[aug_idx], lc_band[aug_idx], aug_idx, mode
        else:
            raise NameError(
                f'mode must be one of ("cutoff", "cuton", "sparse", "full") but got {mode}'
            )

    def normalize_flux(self, flux, fluxerr, flux_norm_info=None):

        if self.log_flux:
            fluxerr = fluxerr / (np.abs(flux) + 1.0) / np.log(10)
            flux = np.sign(flux) * np.log10(np.abs(flux) + 1.0)

        if self.flux_normalization == "max_partial_scalar":
            if flux_norm_info is None:
                scalar = flux.max() + 1e-6
            else:
                scalar = flux_norm_info[3]
            flux = flux / scalar
            fluxerr = fluxerr / scalar
            fluxnorm_info = (
                "log_flux",
                self.log_flux,
                "max_partial_scalar",
                float(scalar),
            )

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
                t_max = self.t_max
            else:
                t_min, t_max = time_norm_info[1], time_norm_info[2]
            time = (time - t_min) / t_max
            return time, ("minmax", float(t_min), float(t_max))
        else:
            return time, ("none")

    def __len__(self):

        return len(self.index_pairs)

    def normalize(self, lc_part, lc_full):

        # Normalize time
        time_part, time_norm_info = self.normalize_time(lc_part[:, 0])
        time_full, _ = self.normalize_time(lc_full[:, 0], time_norm_info=time_norm_info)

        # Normalize flux
        flux_part, flux_err_part, flux_norm_info = self.normalize_flux(
            lc_part[:, 1], lc_part[:, 2]
        )
        flux_full, flux_err_full, _ = self.normalize_flux(
            lc_full[:, 1], lc_full[:, 2], flux_norm_info=flux_norm_info
        )

        return (
            time_part,
            flux_part,
            flux_err_part,
            time_full,
            flux_full,
            flux_err_full,
            time_norm_info,
            flux_norm_info,
        )

    def get_raw_lightcurve(self, idx):

        lc_idx, mode = self.index_pairs[idx]
        lc_full = self.full_light_curve_array[lc_idx].copy()
        band_idx_full = self.band_array[lc_idx] 
        length = self.full_lengths[lc_idx]

        return lc_idx, mode, lc_full, band_idx_full, length

    def get_in_sample_mask(self, length, aug_idx):

        mask = np.zeros(length, dtype=bool)
        mask[aug_idx] = True

        return mask

    def get_padding_mask(self, length, full_length="max"):

        if full_length == "max":
            full_length = self.max_length
        else:
            raise ValueError("Only 'max' is supported for 'length'")

        padding_mask = np.zeros(full_length, dtype=bool)
        padding_mask[:length] = True

        return padding_mask

    def pad(self, arr, padding_mask):

        #print(padding_mask.sum(), arr.shape, padding_mask.shape, len(padding_mask))
        length = len(padding_mask)
        shape = list(arr.shape)
        shape[0] = length
        padded_arr = np.zeros_like(arr, shape=shape)
        #print(padded_arr[padding_mask].shape, arr.shape)
        edge_arr = len(arr) # Do this since arr will have a smaller size
        padded_arr[:edge_arr][padding_mask[:edge_arr]] = arr

        return padded_arr

    def __getitem__(self, idx):
        """
        Get the light curve and preform preprocessing
        (augmentation, flux/time normalization) on the light curve.
        For augmented light curves, return the full light curve as well.
        """
        lc_idx, mode, lc_full, band_idx_full, length = self.get_raw_lightcurve(idx)
        lc_part, band_idx_part, aug_idx, mode = self.apply_augmentation(lc_full, band_idx_full, mode)
        (
            time_part,
            flux_part,
            flux_err_part,
            time_full,
            flux_full,
            flux_err_full,
            time_norm_info,
            flux_norm_info,
        ) = self.normalize(lc_part, lc_full)

        # Mask for in-sample points
        #mask = np.zeros(len(time_full), dtype=bool)
        #mask[aug_idx] = True
        
        aug_mask = self.get_in_sample_mask(length, aug_idx)
        pad_mask = self.get_padding_mask(length)

        


        scalar_f_part = np.full_like(time_part, flux_norm_info[-1])
        x_input = np.vstack([time_part, flux_part, scalar_f_part]).T

        #print(mode, x_input.shape, pad_mask.sum(), length, len(pad_mask), aug_idx)
        #print("-------------------")
        #print("-------------------")


        training_data = {
            "x_input": x_input,
            "time_part": time_part,
            "flux_part": flux_part,
            "flux_err_part": flux_err_part,
            "band_idx_part": band_idx_part,
            "time_full": time_full,
            "flux_full": flux_full,
            "flux_err_full": flux_err_full,
            "band_idx_full": band_idx_full,
            "in_sample_mask": aug_mask,
        }

        meta_data = {
            "peak_time": torch.tensor(float(time_full.max())),
            "peak_flux": torch.tensor(float(flux_full.max())),
            "time_norm_info": time_norm_info,
            "flux_norm_info": flux_norm_info,
            "augmentation": mode,
            "raw_idx": lc_idx,
        }

        training_data = {key: self.pad(value, pad_mask) for key, value in training_data.items()}


        return {
            "x_input": torch.from_numpy(training_data["x_input"]),
            "time_part": torch.from_numpy(training_data["time_part"]).unsqueeze(1),
            "flux_part": torch.from_numpy(training_data["flux_part"]).unsqueeze(1),
            "flux_err_part": torch.from_numpy(training_data["flux_err_part"]).unsqueeze(1),
            "band_idx_part": torch.from_numpy(training_data["band_idx_part"]).unsqueeze(1),
            "time_full": torch.from_numpy(training_data["time_full"]).unsqueeze(1),
            "flux_full": torch.from_numpy(training_data["flux_full"]).unsqueeze(1),
            "flux_err_full": torch.from_numpy(training_data["flux_err_full"]).unsqueeze(1),
            "band_idx_full": torch.from_numpy(training_data["band_idx_full"]).unsqueeze(1),
            "in_sample_mask": torch.from_numpy(training_data["in_sample_mask"]).unsqueeze(1),
            "pad_mask": torch.from_numpy(pad_mask).unsqueeze(1),
            "length": length,
            "peak_time": time_full.max(),
            "peak_flux": flux_full.max(),
            "time_norm_info": time_norm_info,
            "flux_norm_info": flux_norm_info,
            "augmentation": mode,
            "raw_idx": lc_idx,
        }


# ======================================
# Collate Function for Dataloader
# ======================================


def enhanced_collate_fn_(batch):
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
        "band_idx": pad(
            [b["band_idx_part"] for b in batch], (B, max_part, 1), torch.long
        ),
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
        "in_sample_mask_part": pad(
            [torch.ones_like(b["x_input"]) for b in batch],
            (B, max_part, 3),
            torch.bool,
        )[:, :, :1].squeeze(2),
        "pad_mask_full": pad(
            [torch.ones_like(b["time_full"]) for b in batch],
            (B, max_full, 1),
            torch.bool,
        ).squeeze(2),
        "lengths": torch.tensor([len(b["time_full"]) for b in batch]),
        "full_lengths": torch.tensor([len(b["time_full"]) for b in batch]),
        "peak_time": torch.tensor([b["peak_time"] for b in batch]),
        "peak_flux": torch.tensor([b["peak_flux"] for b in batch]),
        "flux_norm_info": [b["flux_norm_info"] for b in batch],
        "time_norm_info": [b["time_norm_info"] for b in batch],
        "raw_idx": [b["raw_idx"] for b in batch],
    }
    return coll

def enhanced_collate_fn(batch):
    """
    Pads a batch of light curves (partial and full) for training.
    """
    B = len(batch)
    max_part = max(len(b["x_input"]) for b in batch)
    max_full = max(len(b["time_full"]) for b in batch)

    coll = {
        "x_input": torch.stack([b["x_input"] for b in batch]),
        "band_idx": torch.stack([b["band_idx_part"] for b in batch]),
        "time_full": torch.stack([b["time_full"] for b in batch]),
        "flux_full": torch.stack([b["flux_full"] for b in batch]),
        "flux_err_full": torch.stack([b["flux_err_full"] for b in batch]),
        "band_idx_full": torch.stack(
            [b["band_idx_full"] for b in batch]),
        "in_sample_mask": torch.stack(
            [b["in_sample_mask"] for b in batch]
        ).squeeze(2),
        "in_sample_mask_part": torch.stack(
            [b["in_sample_mask"] for b in batch]
        ).squeeze(2),
        "pad_mask_full": torch.stack([b["pad_mask"] for b in batch]).squeeze(2),
        "lengths": torch.tensor([b["length"] for b in batch]),
        "full_lengths": torch.tensor([b["length"] for b in batch]),
        "peak_time": torch.tensor([b["peak_time"] for b in batch]),
        "peak_flux": torch.tensor([b["peak_flux"] for b in batch]),
        "flux_norm_info": [b["flux_norm_info"] for b in batch],
        "time_norm_info": [b["time_norm_info"] for b in batch],
        "raw_idx": [b["raw_idx"] for b in batch],
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
        test_run=False,
        t_max=200,
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
        self.test_run = test_run
        self.t_max = t_max

        with open(lightcurve_path, "rb") as f:
            filtered_data = pickle.load(f)

        self.full_light_curves, self.full_class_names = extract_curves_from_filtered_data(
            filtered_data, min_points=min_points, class_filter=class_filter
        )

        if self.test_run:
            self.full_light_curves = self.full_light_curves[:batch_size*100]
            self.full_class_names = self.full_class_names[:batch_size*100]

    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = LightCurvePreprocessor(
                self.full_light_curves,
                self.full_class_names,
                flux_normalization=self.flux_normalization,
                log_flux=self.log_flux,
                time_normalization=self.time_normalization,
                augmentation=self.augmentation,
                num_aug_per_curve=self.num_aug_per_curve,
                min_obs_fraction=self.min_obs_fraction,
                t_max=self.t_max
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
