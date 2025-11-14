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
from TAE.util import fill_config, extract_curves_from_filtered_data, registry, register_module
from multiprocessing import Process, Manager
from TAE.datasets.utils import BAND_IDX_MAP
from TAE.datasets.preprocessing import *
from TAE.datasets.masking import *
from TAE.datasets.lightcurve_preprocessor import LightCurvePreprocessor
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
import pandas as pd
from TAE.datasets.elasticc_utils import ELAsTiCC_to_Astrophysical_mappings

@register_module
class LightCurveDataLoader(pl.LightningDataModule):
    """
    Light curve preprocessor that applies augmentation and normalization to light curves.
    """

    def __init__(
        self,
        lightcurve_path=".",
        class_filter=None,
        min_points=5,
        max_points=100,
        t_max=200,
        batch_size=32,
        augmentations=None,
        num_workers=0,
        mode='pickle',
        include_non_detections=False,
        include_meta_data=False,
        group_classes=True,
        zero_non_detections=False,
        pin_memory=True,
        balance_class_samples=False,
        **kwargs
    ):
        super().__init__()
        
        self.manager = Manager()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_points = min_points
        self.max_points = max_points
        self.t_max = t_max
        self.mode = mode
        self.lightcurve_path = lightcurve_path
        self.class_filter = class_filter
        self.prefetch_factor = 2 if self.num_workers else None
        self.persistent_workers = self.num_workers > 0
        self.include_non_detections = include_non_detections
        self.include_meta_data = include_meta_data
        self.meta_data = None
        self.group_classes = group_classes
        self.zero_non_detections = zero_non_detections
        self.pin_memory = pin_memory
        self.balance_class_samples = balance_class_samples
        if self.zero_non_detections and not include_non_detections:
            import sys
            print("WARNING: zero_non_detections is True but non-detections are not used", file=sys.stderr)
            self.zero_non_detection = True
        if include_non_detections:
            if mode != "parquet":
                import sys
                print("WARNING: include_non_detect only works with mode='parquet', ignoring", file=sys.stderr)
        
        if augmentations is not None:
            augmentations = [registry[aug](min_points=min_points, max_points=max_points) for aug in augmentations]
            kwargs.update({"augmentations": augmentations})

        self.kwargs = kwargs

    def load_data(self):

        if self.mode == 'pickle':
            with open(self.lightcurve_path, "rb") as f:
                filtered_data = pickle.load(f)

            full_light_curves, full_class_names = extract_curves_from_filtered_data(
                filtered_data, min_points=self.min_points, class_filter=self.class_filter
            )
            self.full_light_curves = full_light_curves #self.manager.list(full_light_curves)
            self.full_class_names = full_class_names #self.manager.list(full_class_names)

        elif self.mode == 'parquet':
            df = pd.read_parquet(self.lightcurve_path, columns=['ELASTICC_class','MJD','FLUXCAL','FLUXCALERR','BAND','PHOTFLAG', 'REDSHIFT_FINAL', "SNID"])
            df['class'] = df['ELASTICC_class'].map(ELAsTiCC_to_Astrophysical_mappings)
            if self.class_filter is not None:
                df = df[df['class'].isin(tuple(self.class_filter))]

            detection_flag = 4096
            first_detection_flag = 2048
            selected_flags = detection_flag | first_detection_flag
            saturation_flag = 1024 

            def include_point(f, include_non_detections=self.include_non_detections):
                if include_non_detections:
                    return (f & saturation_flag) ^ saturation_flag
                else:
                    return (f & selected_flags) * ((f & saturation_flag) ^ saturation_flag)
            
            flags = df['PHOTFLAG'].to_numpy()
            df['nobs'] = [int(np.count_nonzero(include_point(np.asarray(f), False).astype(bool))) for f in flags]
            df = df[df['nobs'] >= self.min_points]

            kept_flags = [include_point(np.asarray(f)).astype(bool) for f in df['PHOTFLAG'].to_numpy()]


            MJD_arr      = df['MJD'].to_numpy()
            FLUX_arr     = df['FLUXCAL'].to_numpy()
            FLUXERR_arr  = df['FLUXCALERR'].to_numpy()
            BAND_arr     = df['BAND'].to_numpy()
            PHOTFLAG_arr     = df['PHOTFLAG'].to_numpy()

            df['MJD_clean']        = [np.asarray(m)[m0] for m, m0 in zip(MJD_arr,     kept_flags)]
            df['FLUXCAL_clean']    = [np.asarray(x)[m0] for x,  m0 in zip(FLUX_arr,    kept_flags)]
            df['FLUXCALERR_clean'] = [np.asarray(x)[m0] for x,  m0 in zip(FLUXERR_arr, kept_flags)]
            df['BAND_clean']       = [np.asarray(x)[m0] for x,  m0 in zip(BAND_arr,    kept_flags)]
            df['PHOTFLAG_clean']   = [np.asarray(x)[m0] for x,  m0 in zip(PHOTFLAG_arr,    kept_flags)]
            df['detection_flag_clean']   = [np.asarray(x)[m0] > 0 for x,  m0 in zip(PHOTFLAG_arr,    kept_flags)]

            if not self.group_classes:
                df['class'] = df['ELASTICC_class']

            df = df.drop(columns=['ELASTICC_class','MJD','FLUXCAL','FLUXCALERR','BAND'])

            self.full_light_curves = df
            self.full_class_names = df['class'].to_list() #self.manager.list(df['class'].to_list())


        else:
            raise ValueError(f"Mode ({self.mode}) invalid.")
        
    def setup(self, *args, **kwargs):

        self.load_data()
        self.init_class_weights()
        self.init_light_curves()
        self.init_band_index()
        self.init_meta_data()
        self.init_data_stats()
        

        self.dataset = LightCurvePreprocessor(        
            light_curves = self.full_light_curve_array,
            class_names = self.full_class_names,
            class_weights = self.class_weights,
            band_indices = self.band_array,
            band_weights = self.band_weights,
            flux_stats = self.flux_stats,
            min_points=self.min_points,
            max_points=self.max_points,
            meta_data=self.meta_data,
            **self.kwargs
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
        
    def init_class_weights(self):

        print("Weighting Classes...")
        self.class_label_names = []#self.manager.list()
        self.class_label_names.extend(list(set(self.full_class_names)))

        self.class_map = {}#self.manager.dict()
        self.class_map.update(dict(zip(self.class_label_names, range(len(self.class_label_names)))))
        
        self.label2class = {}#self.manager.dict()
        self.label2class.update(dict(zip(self.class_map.values(), self.class_map.keys())))

        self.class_counts = {}#self.manager.dict()
        self.class_counts.update({key: self.full_class_names.count(key) for key in self.class_map})

        self.class_weights = {}#self.manager.dict()
        self.class_weights.update({key: len(self.full_class_names)/self.class_counts[key]/len(self.class_label_names) for key in self.class_counts})

    def init_meta_data(self):

        if self.include_meta_data:
            assert self.mode == 'parquet', "Only `mode='parquet'` supports meta data"

            columns = ['detection_flag_clean']
            meta_data_array = [
                np.column_stack([np.asarray(self.full_light_curves.iloc[i][c], dtype=np.float32) for c in columns])
                for i in tqdm(range(len(self.full_light_curves)), desc="Extracting PHOT Flags")
            ]
            columns = ['REDSHIFT_FINAL']
            redshift_data_array = [
                np.column_stack([np.asarray(self.full_light_curves.iloc[i][c], dtype=np.float32) for c in columns])
                for i in tqdm(range(len(self.full_light_curves)), desc="Extracting Redshifts")
            ]
            columns = ['SNID']
            snid_data_array = [
                np.column_stack([np.asarray(self.full_light_curves.iloc[i][c], dtype=np.int64) for c in columns])
                for i in tqdm(range(len(self.full_light_curves)), desc="Extracting SNID")
            ]
            self.meta_data = {'detection_flag':meta_data_array}
            self.meta_data.update({'REDSHIFT_FINAL': redshift_data_array})
            self.meta_data.update({'SNID': snid_data_array})

        if self.zero_non_detections:
            if self.include_meta_data:
                for arr, detection_flag in tqdm(zip(self.full_light_curve_array, meta_data_array), desc='Setting Negative Fluxes to Zero'):
                    arr[arr[:, 1]<0, 1] = 0.0
                    arr[detection_flag[:, 0]==0, 1] = 0.0
            else:
                for arr in tqdm(self.full_light_curve_array, desc='Setting Negative Fluxes to Zero'):
                    arr[arr[:, 1]<0, 1] = 0.0
         

    def init_light_curves(self):

        if self.mode == 'parquet':
            columns = ["MJD_clean", "FLUXCAL_clean", "FLUXCALERR_clean"]
            self.full_light_curve_array = [
                np.column_stack([np.asarray(self.full_light_curves.iloc[i][c], dtype=np.float32) for c in columns])
                for i in tqdm(range(len(self.full_light_curves)), desc="Extracting LCs")
            ]
        elif self.mode == 'pickle':
            columns = ["days_since_first_observation", "FLUXCAL", "FLUXCALERR"]
            self.full_light_curve_array = [
                df[columns].astype(np.float32).reset_index(drop=True).values
                for df in tqdm(self.full_light_curves, desc="Extracting LCs") if len(df) >= self.min_points
            ]
        else:
            raise ValueError(f"Mode ({self.mode}) invalid.")

        #if self.zero_non_detections:
        #    for arr in tqdm(self.full_light_curve_array, desc='Setting Negative Fluxes to Zero'):
        #        arr[:, 1][arr[:, 1]<0] = 0.0
        #        
        #        #assert (arr[:, 1] >= 0).all() # BE CAREFUL! MANAGED LISTS ARE NOT REGULARLY MUTABLE
        
        #self.full_light_curve_array = self.manager.list(self.full_light_curve_array)
        
    def init_band_index(self):

        if self.mode == 'pickle':

            self.band_array = [
                df["band" if "band" in df.columns else "filter"]
                .map(BAND_IDX_MAP)
                .reset_index(drop=True)
                .values
                for df in tqdm(self.full_light_curves, desc="Mapping Bands to Indices") if len(df) >= self.min_points
            ]
        elif self.mode == 'parquet':
            self.band_array = [
                np.array([BAND_IDX_MAP[x.lower()] for x in self.full_light_curves.iloc[i]["BAND_clean"]])
                for i in tqdm(range(len(self.full_light_curves)), desc="Mapping Bands to Indices")
            ]

        all_bands = np.hstack(self.band_array)
        bin_counts = np.bincount(all_bands)
        self.band_weights = {} #self.manager.dict()
        self.band_weights.update({key: len(all_bands)/bin_counts[key]/6 for key in range(6)})
        print("Found Band Weighting")

    def init_data_stats(self):

        self.full_lengths = [] #self.manager.list()
        self.full_lengths.extend([len(arr) for arr in tqdm(self.band_array, desc="Getting Lengths")])
        self.max_length = max(self.full_lengths)

        if self.t_max == "infer":
            ts = np.array([lc[:, 0].max()-lc[:, 0].min() for lc in tqdm(self.full_light_curve_array, desc="Getting Time Stats")])
            self.t_max = 2*ts.std()/2 # +/- 1 sigma
            print("Inferred t_max =", self.t_max)
        print("Found Maximum Light Curve Length:", self.max_length)

        # get flux stats
        print("Getting Flux Stats...")
        flat_lcs = np.concatenate(self.full_light_curve_array, axis=0)

        self.flux_stats = {} #self.manager.dict()
        self.flux_stats['min'] = flat_lcs[:, 1].min()
        self.flux_stats['max'] = flat_lcs[:, 1].max()
        self.flux_stats['mean'] = flat_lcs[:, 1].mean()
        self.flux_stats['std'] = flat_lcs[:, 1].std()
        self.flux_stats['t_max'] = self.t_max
        print('Got the Following Flux Stats:')
        for key in self.flux_stats:
            print(key, '|', self.flux_stats[key])
        print(
            f"Dataset contains {len(self.full_light_curves)} samples"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            in_order=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=enhanced_collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=enhanced_collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            drop_last=True,
        )
    
def test():

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # Run a test of the modules
    lightcurve_path = "/projects/ncsa/caps/skai/data/ELAsTiCC/complete.parquet"
    class_filters = ['SNIa', 'SNII', 'SNIb/c', 'SNIax', 'SN91bg', 'SLSN', 'KN', 'CART', 'TDE', 'PISN', 'Dwarf Novae', 'ILOT', 'M-dwarf Flare']
    mode = 'parquet'
    augmentations = ['CutOffSample', 'FullSample', 'BandSample', 'SparseSample', 'CutOnSample']

    dataloader = LightCurveDataLoader(lightcurve_path=lightcurve_path, 
                                      mode=mode, class_filter=class_filters, 
                                      num_workers=1, augmentations=augmentations,
                                      zero_non_detections=True, include_meta_data=True,
                                      include_non_detections=True, group_classes=False, resample=False, 
                                      t_offset=-1.0, flux_normalization='minmax', time_normalization='std',
                                      t_max='infer', pin_memory=True)
    dataloader.setup()

    batch_data = []

    for batch in dataloader.test_dataloader():
        len(batch)
        for i in tqdm(range(len(batch['x_input']))):
            batch_data.append(batch['x_input'][i][batch['in_sample_mask'][i]])
            #print(batch_data[-1].shape)
    batch_data = np.vstack(batch_data)
    for i in range(3):
        name = ['Scaled Time', 'Scaled Flux', 'Scaled Flux Error'][i]
        plt.figure()
        plt.hist(batch_data[:, i], bins=int(np.sqrt(len(batch_data))), histtype='step')
        plt.xlabel(name)
        plt.ylabel('counts')
        #plt.yscale('log')
        plt.savefig(f'figs/tests/test_dataloader_hist_{name}.png')
        


    light_curve = batch['x_input'].detach().cpu().numpy()
    print('Batch Shape:', light_curve.shape[0])

    
    for i in tqdm(range(light_curve.shape[0])):
        band_idx = batch['band_idx'][i, :, 0].detach().cpu().numpy()
        index = batch['raw_idx'][i]
        raw_light_curve = dataloader.dataset.light_curves[index].copy()[:dataloader.dataset.max_points]
        lc = light_curve[i]
        lc[:, 1], lc[:, 2] = dataloader.dataset.flux_normalizer.inverse_transform(lc[:, 1], error=lc[:, 2])
        lc[:, 1], lc[:, 2] = dataloader.dataset.semilog_transform.inverse_transform(lc[:, 1], error=lc[:, 2])
        plt.figure()
        plt.scatter(lc[:, 0], lc[:, 1], c=band_idx, label='Untransformed')
        plt.scatter(lc[:len(raw_light_curve), 0], raw_light_curve[:, 1], c='k', label='Original', marker='x')
        plt.legend()
        plt.savefig(f'figs/tests/test_dataloader_{i}.png')
        plt.close("all")