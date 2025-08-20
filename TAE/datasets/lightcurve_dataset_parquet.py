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
from TAE.util import register_module
import pandas as pd
from TAE.datasets.elasticc_utils import ELAsTiCC_to_Astrophysical_mappings

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
        resample=False,
    ):
        self.full_light_curves = full_light_curves
        self.full_class_names = list(full_class_names)
        self.flux_normalization = flux_normalization
        self.log_flux = log_flux
        self.time_normalization = time_normalization
        self.augmentation = augmentation
        self.min_points = min_points
        self.min_obs_fraction = min_obs_fraction
        self.t_max = t_max
        self.resample = resample

        print("Weighting Classes...")
        self.class_label_names = set(self.full_class_names)
        self.class_map = dict(zip(self.class_label_names, range(len(self.class_label_names))))
        self.label2class = dict(zip(self.class_map.values(), self.class_map.keys()))
        self.class_counts = {key: self.full_class_names.count(key) for key in self.class_map}
        self.class_weights = {key: len(self.full_class_names)/self.class_counts[key]/len(self.class_label_names) for key in self.class_counts}

        print("Building Index...")

        # Build (light curve index, augmentation mode) list
        self.index_pairs = []
        for i in range(len(self.full_light_curves)):
            if self.full_light_curves['nobs'].iloc[i] < min_points:
                continue
            self.index_pairs.append((i, self.augmentation))
            #for _ in range(num_aug_per_curve):
            #    if self.augmentation == "mixed":
            #        mode = np.random.choice(["cutoff", "cuton", "sparse", "full"])
            #    else:
            #        mode = self.augmentation
            #    self.index_pairs.append((i, mode))

        columns = ["MJD_clean", "FLUXCAL_clean", "FLUXCALERR_clean"]

        self.full_light_curve_array = [
            np.column_stack([np.asarray(self.full_light_curves.iloc[i][c], dtype=np.float32) for c in columns])
            for i in tqdm(range(len(self.full_light_curves)), desc="Extracting LCs")
        ]

        for arr in tqdm(self.full_light_curve_array, desc='Setting Negative Fluxes to Zero'):
            arr[:, 1][arr[:, 1]<0] = 0.0

        self.band_array = [
            np.array([BAND_IDX_MAP[x.lower()] for x in self.full_light_curves.iloc[i]["BAND_clean"]])
            for i in tqdm(range(len(self.full_light_curves)), desc="Mapping Bands to Indices")
        ]

        self.full_lengths = self.full_light_curves['nobs'].astype(int).values
        self.max_length = max(self.full_lengths)

        if self.t_max == "infer":
            ts = np.array([lc[:, 0].max()-lc[:, 0].min() for lc in tqdm(self.full_light_curve_array, desc="Getting Time Stats")])
            self.t_max = 2*ts.std() # +/- 1 sigma
            print("Inferred t_max =", self.t_max)
        print("Found Maximum Light Curve Length:", self.max_length)

        # get flux stats
        print("Getting Flux Stats...")
        flat_lcs = np.concatenate(self.full_light_curve_array, axis=0)
        if self.log_flux:
            flat_lcs[:, 1] = np.sign(flat_lcs[:, 1]) * np.log10(np.abs(flat_lcs[:, 1]) + 1.0)
            assert (flat_lcs[:, 1] >= 0).all()
        self.flux_stats = {}
        self.flux_stats['min'] = flat_lcs[:, 1].min()
        self.flux_stats['max'] = flat_lcs[:, 1].max()
        self.flux_stats['mean'] = flat_lcs[:, 1].mean()
        self.flux_stats['std'] = flat_lcs[:, 1].std()
        print('Got the Following Flux Stats:')
        for key in self.flux_stats:
            print(key, '|', self.flux_stats[key])

        print(
            f"Dataset contains {len(self.index_pairs)} samples from {len(full_light_curves)} full curves."
        )

    def apply_augmentation(self, lc, lc_band, mode):
        """
        Apply a single augmentation (cutoff / cuton / sparse / full) to
        one light curve.  Guarantees that the returned segment has at least
        `min_len` points; if that is impossible, falls back to returning the
        full curve.
        """
        n = len(lc)
        peak_idx = lc[:, 1].argmax()            # column-1 is log-flux

        # Decide the augmentation mode
        if self.augmentation == "mixed":
            mode = np.random.choice(["cutoff", "sparse"]) # "cuton", "sparse",
        else:
            mode = self.augmentation

        # Minimum allowed length for any augmented curve
        min_len = max(self.min_points, int(self.min_obs_fraction * n))

        # ------------------------------------------------------------------ #
        # 1) CUTOFF  – keep pre-peak part
        # ------------------------------------------------------------------ #
        if mode == "cutoff":
            # 80 % chance to cut before the peak, 20 % anywhere
            if np.random.rand() < 0.95:
                hi = max(min(peak_idx, n - 1), min_len)
            else:
                hi = n

            # If hi == min_len we cannot draw a randint range → fall back
            if hi <= min_len:
                # Fall back to returning the full curve (mode='full')
                return lc, lc_band, np.arange(n), "full"

            aug_idx = np.random.randint(min_len, hi)
            return lc[:aug_idx], lc_band[:aug_idx], np.arange(aug_idx), mode

        # ------------------------------------------------------------------ #
        # 2) CUTON  – keep post-peak part
        # ------------------------------------------------------------------ #
        if mode == "cuton":
            hi = n - min_len
            if hi <= 0:
                return lc, lc_band, np.arange(n), "full"
            aug_idx = np.random.randint(0, hi)
            return lc[aug_idx:], lc_band[aug_idx:], np.arange(aug_idx, n), mode

        if mode == "cuton":
            hi = n - min_len
            aug_idx = np.random.randint(0, hi)
            return lc[aug_idx:], lc_band[aug_idx:], np.arange(aug_idx, n), mode
        if mode == "sparse":
            k = np.random.randint(min_len, n)
            aug_idx = np.sort(np.random.choice(n, k, replace=False))
            return lc[aug_idx], lc_band[aug_idx], aug_idx, mode
        if mode == "full":
            aug_idx = np.arange(n)
            return lc, lc_band, aug_idx, mode
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

        elif self.flux_normalization == "zscore":
            if flux_norm_info is None:
                mu = flux.mean()
                sd = flux.std() or 1.0
            else:
                mu, sd = flux_norm_info[3], flux_norm_info[4]
            flux = (flux - mu) / sd
            fluxerr = fluxerr / sd
            fluxnorm_info = ("log_flux", self.log_flux, "zscore", float(mu), float(sd))

        elif self.flux_normalization == 'minmax':
            scale = self.flux_stats['max'] - self.flux_stats['min']
            flux = (flux - self.flux_stats['min']) / (scale) - 0.5
            fluxerr = fluxerr / scale
            fluxnorm_info = ("log_flux", self.log_flux, "minmax", float(self.flux_stats['min']), float(scale))

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
        class_name = self.full_class_names[lc_idx]
        lc_full = self.full_light_curve_array[lc_idx].copy()
        band_idx_full = self.band_array[lc_idx] 
        length = self.full_lengths[lc_idx]

        return lc_idx, mode, lc_full, band_idx_full, length, class_name

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
        #padded_arr[:edge_arr][padding_mask[:edge_arr]] = arr
        # safer choice
        valid_rows = padding_mask[:edge_arr]
        padded_arr[:edge_arr][np.where(valid_rows)[0]] = arr

        return padded_arr

    def preprocess(self, raw_idx, lc_full, band_idx_full, length, lc_part, band_idx_part, aug_idx, mode, class_name):

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
        ivar = 1/((flux_err_part)**2+1e-4)*1e-4 # the 1e-4 is a hyperparameter
        x_input = np.vstack([time_part, flux_part, scalar_f_part, ivar]).T

        scalar_f_full = np.full_like(time_full, flux_norm_info[-1])
        ivar_full = 1/((flux_err_full)**2+1e-4)*1e-4 # the 1e-4 is a hyperparameter
        x_input_full = np.vstack([time_full, flux_full, scalar_f_full, ivar_full]).T

        #print(mode, x_input.shape, pad_mask.sum(), length, len(pad_mask), aug_idx)
        #print("-------------------")
        #print("-------------------")


        training_data = {
            "x_input": x_input,
            "x_input_full": x_input_full,
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
            #"peak_time": torch.tensor(float(time_full.max())),
            #"peak_flux": torch.tensor(float(flux_full.max())),
            "time_norm_info": time_norm_info,
            "flux_norm_info": flux_norm_info,
            "augmentation": mode,
            "raw_idx": raw_idx,
        }

        training_data = {key: self.pad(value, pad_mask) for key, value in training_data.items()}


        return {
            "x_input": torch.from_numpy(training_data["x_input"]),
            "x_input_full": torch.from_numpy(training_data["x_input_full"]),
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
            #"peak_time": time_full.max(),
            #"peak_flux": flux_full.max(),
            "time_norm_info": time_norm_info,
            "flux_norm_info": flux_norm_info,
            "augmentation": mode,
            "raw_idx": raw_idx,
            "class_name": class_name,
            "class_weight": self.class_weights[class_name]
        }



    def __getitem__(self, idx):
        """
        Get the light curve and preform preprocessing
        (augmentation, flux/time normalization) on the light curve.
        For augmented light curves, return the full light curve as well.
        """
        lc_idx, mode, lc_full, band_idx_full, length, class_name = self.get_raw_lightcurve(idx)
        if self.resample:
            lc_full[:, 1] = np.random.normal(loc=lc_full[:, 1], scale=lc_full[:, 2]).astype(lc_full.dtype)
            lc_full[:, 1][lc_full[:, 1] < 0] = 0.0
        lc_part, band_idx_part, aug_idx, mode = self.apply_augmentation(lc_full, band_idx_full, mode)
        return self.preprocess(idx, lc_full, band_idx_full, length, lc_part, band_idx_part, aug_idx, mode, class_name)
        


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

    for i in range(len(batch)):
        b = batch[i]
        b['x_input'][~b['in_sample_mask'][:, 0]] = b['x_input'][b['in_sample_mask'][:, 0]][-1] # Forward fill
    x_input = torch.stack([b["x_input"] for b in batch])
    time = x_input[:, :, 0] # (B, S)
    mask = torch.stack(
            [b["in_sample_mask"] for b in batch]
        ).squeeze(2) # (B, S)
    time_sorted = None #torch.sort(torch.unique(torch.ravel(time[mask])), dim=0, descending=True)[0] # (T)
    sequence_batch_mask = None #torch.stack([((time == t0) & mask) for t0 in time_sorted]) #(T, B, S)

    coll = {
        "x_input": torch.stack([b["x_input"] for b in batch]),
        "x_input_full": torch.stack([b["x_input_full"] for b in batch]),
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
        #"peak_time": torch.tensor([b["peak_time"] for b in batch]),
        #"peak_flux": torch.tensor([b["peak_flux"] for b in batch]),
        "flux_norm_info": [b["flux_norm_info"] for b in batch],
        "time_norm_info": [b["time_norm_info"] for b in batch],
        "raw_idx": [b["raw_idx"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
        "class_weight": torch.tensor([b["class_weight"] for b in batch]),
        "time_sorted": time_sorted,
        "sequence_batch_mask": sequence_batch_mask,
    }
    return coll


# ======================================
# PyTorch Lightning DataModule
# ======================================

@register_module
class SNDataModuleParquet(pl.LightningDataModule):
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
        resample=False,
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
        self.resample = resample

    def setup(self, stage=None):

        # read in the file
        df = pd.read_parquet(self.lightcurve_path, columns=['ELASTICC_class','MJD','FLUXCAL','FLUXCALERR','BAND','PHOTFLAG'])
        ## perform cleaning
        print('Cleaning the data...')
        #print(self.class_filter)

        df['class'] = df['ELASTICC_class'].map(ELAsTiCC_to_Astrophysical_mappings)
        df = df[df['class'].isin(tuple(self.class_filter))]
        
        flags = df['PHOTFLAG'].to_numpy()
        df['nobs'] = [int(np.count_nonzero(np.asarray(f) == 4096)) for f in flags]
        df = df[df['nobs'] >= self.min_points]

        kept_flags = [np.asarray(f) == 4096 for f in df['PHOTFLAG'].to_numpy()]
        MJD_arr      = df['MJD'].to_numpy()
        FLUX_arr     = df['FLUXCAL'].to_numpy()
        FLUXERR_arr  = df['FLUXCALERR'].to_numpy()
        BAND_arr     = df['BAND'].to_numpy()

        df['MJD_clean']        = [np.asarray(m)[m0] for m, m0 in zip(MJD_arr,     kept_flags)]
        df['FLUXCAL_clean']    = [np.asarray(x)[m0] for x,  m0 in zip(FLUX_arr,    kept_flags)]
        df['FLUXCALERR_clean'] = [np.asarray(x)[m0] for x,  m0 in zip(FLUXERR_arr, kept_flags)]
        df['BAND_clean']       = [np.asarray(x)[m0] for x,  m0 in zip(BAND_arr,    kept_flags)]

        df = df.drop(columns=['ELASTICC_class','MJD','FLUXCAL','FLUXCALERR','BAND'])

        self.full_light_curves = df
        self.full_class_names = df['class']

        if self.test_run:
            self.full_light_curves = self.full_light_curves[:self.batch_size*100]
            self.full_class_names = self.full_class_names[:self.batch_size*100]

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
                t_max=self.t_max,
                resample=self.resample,
                min_points=self.min_points
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
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,  
            in_order=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            prefetch_factor=10,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            prefetch_factor=10,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )
