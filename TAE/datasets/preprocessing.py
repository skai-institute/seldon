'''
Module for various pre-processing classes for the light curve datasets
'''

import numpy as np
from functools import reduce


class IndexAugmentation:

    def __init__(self):
        pass

    def transform(self, *args):
        '''returns back the indexes for augmentation'''
        pass

    @classmethod
    def intersect(cls, augmentation_index_a, augmentation_index_b):
        '''Merges two augmentation indexes together and returns the sorted set of both'''
        return np.intersect1d(augmentation_index_a, augmentation_index_b)

    @classmethod
    def full(cls, data):

        return np.arange(len(data))

class SparseSample(IndexAugmentation):


    def __init__(self, min_length=1):
        super().__init__()
        self.min_length = min_length

    def transform(self, data, *args):

        N = len(data)
        if N <= self.min_length:
            return self.full(data)
        n_samples = np.random.randint(self.min_length, N+1)
        augmentation_index = np.random.choice(N, n_samples, replace=False)
        return augmentation_index

class CutOffSample(IndexAugmentation):

    def __init__(self, min_length=1):
        super().__init__()
        self.min_length = min_length

    def transform(data, *args):

        N = len(data)
        if N <= self.min_length:
            return self.full(data)
        n_samples = np.random.randint(self.min_length, N+1)
        augmentation_index = np.arange(n_samples)
        return augmentation_index

class FullSample(IndexAugmentation):

    def __init__(self, min_points=1):
        super().__init__()
        self.min_points = min_points

    def transform(self, data, *args):

        return self.full(data)

class BandSample(IndexAugmentation):

    def __init__(self, min_points=1):
        super().__init__()
        self.min_points = min_points
        self.bands = np.arange(6)

    def transform(self, data, band_index, *args):

        masks = band_idx[None, :] == self.bands[:, None]
        totals = (masks).sum(axis=1)
        valid_total_positions = np.where(totals >= self.min_points)[0]
        if not len(valid_total_positions):
            return self.full(data)
        num_keep = np.random.randint(1, len(valid_total_positions)+1)
        keep = np.random.choice(len(valid_total_positions), num_keep, replace=False)
        augmentation_index = np.where(masks[keep].any(axis=1))[0]
        return augmentation_index

class IndexAugmentationPipeline(IndexAugmentation):

    def __init__(self, min_points=1, max_points=100, *augmentations): 
        '''Augmentations is a list of augmentation classes'''

        self.augmentations = augmentations
        self.min_points = min_points

    def transform(self, data, *args):
        ''' could make this recursively apply random augmentations until the data gets too small'''
        augmentation_indices = []
        augmentation = np.random.choice(self.augmentations)
        augmentation_index = augmentation(data, *args)
        if self.max_points > len(augmentation_index) >= self.min_points:
            extra_augmentation_index = self.transform(data, *args)
            merged_augmentation_index = self.merge(augmentation_index, extra_augmentation_index)
            if len(merged_augmentation_index) > self.min_points:
                augmentation_index = merged_augmentation_index
        return augmentation_index