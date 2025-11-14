# import matplotlib.pyplot as plt
import os
from operator import itemgetter

import filelock
import sys
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import scipy.io as sio
import sklearn.datasets
import torch
import torchvision
import torchvision.transforms as transforms
from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset, random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import glob


# TODO: these will slow down the polyaxon
from tqdm import tqdm


class RandomData(pl.LightningDataModule):
    def __init__(
        self,
        x_dim=101,
        n_samples=1000,
        y_dim=1,
        batch_size=32,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_samples = n_samples
        self.batch_size = batch_size

    def setup(self):

        X_ = torch.tensor(
            np.random.uniform(0, 1, [self.n_samples, self.x_dim]),
            dtype=torch.float,
        )
        Y_ = torch.tensor(
            np.random.uniform(0, 1, [self.n_samples, self.y_dim]),
            dtype=torch.float,
        )
        self.train = TensorDataset(X_, Y_)
        self.valid = TensorDataset(X_, Y_)
        self.teste = TensorDataset(X_, Y_)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.teste, batch_size=1)
