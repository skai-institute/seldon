import pickle
import numpy as np
import torch
from collections import defaultdict
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch
from enum import Enum
from TAE.util import fill_config, extract_curves_from_filtered_data
from multiprocessing import Process, Manager
from TAE.datasets.utils import BAND_IDX_MAP
from TAE.datasets.preprocessing import *
from TAE.datasets.masking import *
from TAE.util import register_module

class DataCache:

    def __init__(self, **data_caches):

        self.data_caches = data_caches
        self.data_names = data_names

    def __getattr__(self, attr):

        if attr in self.data_names:
            return self.data_names[attr]
        else:
            return super().__getattr__(attr)

    def __getitem__(self, index):

        return {key: value[index] for key, value in self.data_caches.items()}



@register_module
class LightCurvePreprocessor(Dataset):
    """
    Light curve preprocessor that applies augmentation and normalization to light curves.
    """

    def __init__(
        self,
        light_curves,
        class_names,
        class_weights,
        band_indices,
        band_weights,
        flux_stats,
        meta_data=None,
        flux_normalization="minmax",
        log_flux=True,
        time_normalization="std",
        augmentations=None,
        min_points=5,
        max_length=400,
        min_obs_fraction=0.5,
        resample=False,
        max_points=50,
        t_offset=0.0,
        force_zero_non_detections=False, # when resampling, keep non-detections as zero
        gumbel_transform=False,
        cache_normalized=False,
        **kwargs,
    ):
        if len(kwargs):
            import sys
            print(f"WARNING: Extra kwargs passed but not used: {kwargs}", file=sys.stderr)
        
        self.max_length = max_length # Max length of the observed lightcurve to fit in the loss
        self.light_curves = light_curves
        self.class_weights = class_weights
        self.class_names = class_names
        self.band_indices = band_indices
        self.band_weights = band_weights
        self.cache_normalized = cache_normalized

        self.meta_data = meta_data
        
        self.t_offset = t_offset
        self.flux_normalization = flux_normalization
        self.log_flux = log_flux
        self.time_normalization = time_normalization
        self.augmentations = augmentations
        self.min_points = min_points
        self.min_obs_fraction = min_obs_fraction
        self.flux_stats = flux_stats
        self.resample = resample
        self.max_points = max_points
        self.force_zero_non_detections = force_zero_non_detections

        np.random.seed(42)

        if self.augmentations is not None:
            self.augmenter = IndexAugmentationPipeline(*self.augmentations, min_points=min_points, max_points=max_points)
        else:
            self.augmenter = FullSample(min_points=min_points, max_points=max_points)

        self.augmentation_masker = AugmentationIndexMask(max_length)
        self.padding_masker = PaddingMask(max_length)

        self.transforms = []

        if gumbel_transform:
            self.gumbel_transform = GumbelTransform(**kwargs)
            self.transforms.append(self.gumbel_transform)

        if self.log_flux:
            self.semilog_transform = SemilogTransform()
            max_flux = self.semilog_transform.transform(self.flux_stats['max'])
            self.transforms.append(self.semilog_transform)
            print(f"Log Flux is True, max log: {max_flux}")
        else:
            max_flux = self.flux_stats['max']

        if self.flux_normalization == "minmax":
            self.flux_normalizer = MaxTransform(max_value=max_flux)
            self.transforms.append(self.flux_normalizer)
        self.transforms = TransformPipeline(*self.transforms)

        if self.cache_normalized:
            lengths = list(map(len, self.light_curves))
            max_light_curve_length = max(lengths)
            data_packages = []
            for i in tqdm(range(len(self.light_curves)), desc='Caching...'):
                package = {}
                lc = self.light_curves[i]
                time = lc[:, 0]
                order = np.argsort(time)
                time = time[order]
                flux = lc[order, 1]
                flux_err = lc[order, 2]
                bands = self.band_indices[i][order]
                flux_norm, flux_err_norm = self.transform(flux, flux_err)
                length = lengths[i]
                padding = max_light_curve_length - length
                package['time'] = np.pad(time, ((0, pad_length),), mode='edge')
                package['flux_norm'] = np.pad(flux_norm, ((0, pad_length),))
                package['flux_err_norm'] = np.pad(flux_err_norm, ((0, pad_length),))
                package['length'] = length
                if self.meta_data is not None:
                    meta_data = self.meta_data['detection_flag'][i][order]
                    package['meta_data'] = np.pad(meta_data, ((0, pad_length),))

                data_packages.append(package)
            data_cache = {key: np.stack([data_packages[i][key] for i in len(self)]) for key in data_cache[0]}
            self.data_cache = DataCache(**data_cache)


    def __len__(self):

        return len(self.light_curves)
    
    def transform(self, data, error=None):

        return self.transforms.transform(data, error)
    
    def inverse_transform(self, data, error=None):

        return self.transforms.inverse_transform(data, error)

    def get(self, index):
        
        # Access the variables
        window = len(self.light_curves[index]) - self.max_length
        if window > 0:
            start = np.random.randint(window)
            end = self.max_length + start
        else:
            start = 0
            end = self.max_length
        light_curve = self.light_curves[index].copy()[start:end] # TODO: Handle This Better
        order = np.argsort(light_curve[:, 0])
        light_curve = light_curve[order]
        band_indices = self.band_indices[index][start:end][order]
        band_weights = np.array(list(map(self.band_weights.get, band_indices)))
        class_name = self.class_names[index]
        class_weight = self.class_weights[class_name]

        # extract metadata (currently just detection flag) and append to the light curve for optional encoding
        if self.meta_data is not None:
            meta_data = self.meta_data['detection_flag'][index][start:end][order]
            light_curve = np.hstack((light_curve, meta_data.astype(np.float32))) # can add conditioning on non-detections

        # Resample the light curve
        if self.resample:
            light_curve[:, 1] = light_curve[:, 1] + np.random.randn(*light_curve[:, 1].shape) * light_curve[:, 2]
            if self.meta_data is not None and self.force_zero_non_detections:
                light_curve[meta_data[:, 0]==0.0, 1] = 0.0

        # Process Flux
        light_curve[:, 1], light_curve[:, 2] = self.transform(light_curve[:, 1], error=light_curve[:, 2])

        # Augmentation and Masking
        augmentation_index = self.augmenter.transform(light_curve, band_indices)
        # If non-detections are included, make sure we have min_points detections
        if self.meta_data is not None: 
            if meta_data[augmentation_index, 0].sum() < self.min_points:
                augmentation_index = np.union1d(augmentation_index, np.where(meta_data[:, 0])[0][:self.min_points])
                augmentation_index = augmentation_index[:self.max_points]

        assert len(augmentation_index) >= self.min_points, f"{self.min_points}, {augmentation_index}"
        augmentation_mask = self.augmentation_masker.transform(light_curve, augmentation_index)
        padding_mask = self.padding_masker.transform(light_curve)

        # Process Time (Do we augment or not?)
        t0 = light_curve[augmentation_mask[:len(light_curve)]][:, 0].min()
        light_curve[:, 0] -= t0 #light_curve[:, 0].min()
        if self.time_normalization == "std":
            light_curve[:, 0] /= self.flux_stats['t_max']
            light_curve[:, 0] += self.t_offset

        pad_length = self.max_length - len(light_curve)
        if pad_length <=0:
            pad_length = 0
            #print(pad_length)
            #import pdb; pdb.set_trace()
        output = {
            "light_curve": torch.from_numpy(np.pad(light_curve, ((0, pad_length), (0,0)), mode='edge')),
            "band_indices": torch.from_numpy(np.pad(band_indices, (0, pad_length))),
            "augmentation_mask": torch.from_numpy(np.pad(augmentation_mask, (0, self.max_length-len(augmentation_mask)))),
            "padding_mask": torch.from_numpy(padding_mask),
            "index": index,
            "class_name": class_name,
            "class_weight": class_weight,
            "band_weights": torch.from_numpy(np.pad(band_weights, (0, pad_length))),
            "scale": self.flux_normalizer.max_value,
        }

        output.update({'meta_data':None, 'redshift':None})
        if self.meta_data is not None:
            output.update({"meta_data": torch.from_numpy(np.pad(meta_data, ((0, pad_length), (0, 0))))})
            if 'REDSHIFT_FINAL' in self.meta_data:
                output.update({'redshift': self.meta_data['REDSHIFT_FINAL'][index]})

        

        light_curve_part = light_curve[augmentation_mask[: len(light_curve)]]
        pad_length = self.max_points - len(light_curve_part) # note: max_points here vs max_length above
        if pad_length <=0:
            pad_length = 0
        output['light_curve_part'] = torch.from_numpy(np.pad(light_curve_part, ((0, pad_length), (0,0)), mode='edge'))
        output['pad_mask_part'] = torch.from_numpy(np.pad(augmentation_mask[augmentation_mask], (0, self.max_points-len(augmentation_mask[augmentation_mask]))))
        output['band_idx_part'] = torch.from_numpy(np.pad(band_indices[augmentation_mask[: len(light_curve)]], (0, pad_length)))

        return output

    def __getitem__(self, index):

        #example = self.legacy_get(index)
        #import pdb; pdb.set_trace()

        return self.legacy_get(index)

    def legacy_get(self, index):

        training_data = self.get(index)

        return {
            "x_input": training_data["light_curve_part"],
            "x_input_full": training_data["light_curve"],
            "time_part": training_data["light_curve_part"][:, 0].unsqueeze(1),
            "flux_part": training_data["light_curve_part"][:, 1].unsqueeze(1),
            "flux_err_part": training_data["light_curve_part"][:, 2].unsqueeze(1),
            "band_idx_part": training_data["band_idx_part"].unsqueeze(1),
            "time_full": training_data["light_curve"][:, 0].unsqueeze(1),
            "flux_full": training_data["light_curve"][:, 1].unsqueeze(1),
            "flux_err_full": training_data["light_curve"][:, 2].unsqueeze(1),
            "band_idx_full": training_data["band_indices"].unsqueeze(1),
            "in_sample_mask": training_data["augmentation_mask"].unsqueeze(1),
            "pad_mask": training_data['padding_mask'].unsqueeze(1),
            "pad_mask_part": training_data['pad_mask_part'].unsqueeze(1),
            "length": training_data['padding_mask'].sum(),
            "peak_time": training_data['light_curve'][:, 0].max(),
            "peak_flux": training_data['light_curve'][:, 1].max(),
            "raw_idx": training_data['index'],
            "class_name": training_data['class_name'],
            "class_weight": training_data['class_weight'],
            "band_weight": training_data['band_weights'],
            "flux_norm_info": ['log', 'log', 'minmax', training_data["scale"]],
            "time_norm_info": None,
            "meta_data": training_data['meta_data'],
            "redshift": training_data['redshift']
        }


        
  