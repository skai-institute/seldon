import pickle
import numpy as np
import torch

import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class LightCurveDataset(pl.LightningDataModule):

    pkl_files = ['x_ts', 'x_static', 'y', 'lengths']

    def __init__(self, data_dir : str, pad_length : int=100, batch_size : int = 100, seed=None, num_workers : int = 4):
        """Loads and processes the dataset."""
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = 100
        self.pad_length = pad_length 
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        for feature in self.pkl_files:
            print("Loading", f'{self.data_dir}/{feature}.pkl')
            with open(f'{self.data_dir}/{feature}.pkl', 'rb') as f:
                setattr(self, feature, pickle.load(f))

        self.x_ts = self.process_time_series(self.x_ts, self.lengths)
        self.x_static = self.process_static_features(self.x_static)
        self.y = np.array(self.y, np.float32)
        self.outlier_mask = np.zeros(len(self.y), dtype=bool)

        N = len(self)
        
        indices = np.arange(N)
        np.random.shuffle(indices)
        edges = (int(0.8*N), int(0.9*N), int(1*N))

        index_train, index_val, index_test = (indices[edges[0]:edges[1]], 
                                              indices[edges[1]:edges[2]], 
                                              indices[edges[2]:]
        )

        self.train = TensorDataset(*self[index_train])
        self.valid = TensorDataset(*self[index_val])
        self.test = TensorDataset(*self[index_test])

    def __getitem__(self, idx):
        return (torch.tensor(self.x_ts[idx], dtype=torch.float32),
                torch.tensor(self.masks[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.outlier_mask[idx], dtype=torch.bool))  


    def process_time_series(self, x_ts, lengths):
        """Converts list of DataFrames to NumPy and pads/truncates sequences."""
        processed_ts = []
        masks = []  # Add mask handling
        for i, df in enumerate(x_ts):
            ts_array = df.to_numpy(dtype=np.float32)
            length = lengths[i]
            if length < self.pad_length:
                # Create mask for valid values (1.0 for valid, 0.0 for padding)
                mask = np.pad(np.ones((length, ts_array.shape[1]), dtype=np.float32),
                              ((0, self.pad_length - length), (0, 0)), mode='constant', constant_values=0)
                # Pad data with zeros
                padded = np.pad(ts_array, ((0, self.pad_length - length), (0, 0)), mode='constant', constant_values=0)
            else:
                mask = np.ones((self.pad_length, ts_array.shape[1]), dtype=np.float32)
                padded = ts_array[:self.pad_length]

            processed_ts.append(padded)
            masks.append(mask)

        self.masks = np.array(masks, dtype=np.float32)
        return np.array(processed_ts, dtype=np.float32)

    def process_static_features(self, x_static):
        """Converts list of OrderedDicts to NumPy array."""
        return np.array([list(entry.values()) for entry in x_static], dtype=np.float32)

    def __len__(self):
        return len(self.x_ts)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1)
