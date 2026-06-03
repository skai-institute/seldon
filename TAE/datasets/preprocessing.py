'''
Module for various pre-processing classes for the light curve datasets
'''

import numpy as np
import torch
import torch.distributions as tdist
from functools import reduce
from TAE.util import register_module


class IndexAugmentation:

    def __init__(self, min_points=0, max_points=0):
        self.min_points = min_points
        self.max_points = max_points

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

@register_module
class SparseSample(IndexAugmentation):


    def __init__(self, min_points=1, **kwargs):
        super().__init__(min_points=min_points, **kwargs)

    def transform(self, data, *args):

        N = len(data)
        if N <= self.min_points:
            return self.full(data)
        n_samples = np.random.randint(self.min_points, N+1)
        augmentation_index = np.random.choice(N, n_samples, replace=False)
        return augmentation_index

@register_module
class CutOffSample(IndexAugmentation):

    def __init__(self, min_points=1, **kwargs):
        super().__init__(min_points=min_points, **kwargs)

    def transform(self, data, *args):

        N = len(data)
        if N <= self.min_points:
            return self.full(data)
        n_samples = np.random.randint(self.min_points, N+1)
        augmentation_index = np.arange(n_samples)
        return augmentation_index
    
@register_module
class CutOnSample(IndexAugmentation):

    def __init__(self, min_points=1, **kwargs):
        super().__init__(min_points=min_points, **kwargs)

    def transform(self, data, *args):

        N = len(data)
        if N <= self.min_points:
            return self.full(data)
        n_samples = np.random.randint(0, N-self.min_points)
        augmentation_index = np.arange(n_samples, N)
        return augmentation_index

@register_module
class FullSample(IndexAugmentation):

    def __init__(self, min_points=1, max_points=100):
        super().__init__(min_points=min_points, max_points=max_points)

    def transform(self, data, *args):

        return self.full(data)[:self.max_points]

@register_module
class BandSample(IndexAugmentation):

    def __init__(self, min_points=1, num_bands=6, **kwargs):
        super().__init__(min_points=min_points, **kwargs)
        self.num_bands = num_bands
        self.bands = np.arange(self.num_bands)

    def transform(self, data, band_index, *args):

        masks = band_index[None, :] == self.bands[:, None]
        totals = (masks).sum(axis=1)
        valid_total_positions = np.where(totals >= self.min_points)[0]
        if not len(valid_total_positions):
            return self.full(data)
        num_keep = np.random.randint(1, len(valid_total_positions)+1)
        keep = valid_total_positions[np.random.choice(len(valid_total_positions), num_keep, replace=False)]
        augmentation_index = np.where(masks[keep].any(axis=0))[0]
        return augmentation_index

class IndexAugmentationPipeline(IndexAugmentation):

    def __init__(self, *augmentations, min_points=1, max_points=100): 
        '''Augmentations is a list of augmentation classes'''
        super().__init__(min_points=min_points, max_points=max_points)
        self.augmentations = augmentations

    def transform(self, data, *args):
        ''' could make this recursively apply random augmentations until the data gets too small'''
        augmentation = np.random.choice(self.augmentations)
        augmentation_index = augmentation.transform(data, *args)
        if len(augmentation_index) < self.min_points:
            import pdb; pdb.set_trace()
        while (len(augmentation_index) > self.max_points) or (np.random.random() < 0.5):
            augmentation = np.random.choice(self.augmentations)
            extra_augmentation_index = augmentation.transform(data, *args)
            merged_augmentation_index = self.intersect(augmentation_index, extra_augmentation_index)
            if len(merged_augmentation_index) >= self.min_points:
                augmentation_index = merged_augmentation_index
        return augmentation_index
    

class GumbelTransform:

    def __init__(self, mu, beta):

        self.mu = mu
        self.beta = beta
        self.dist = tdist.Gumbel(loc=mu, scale=beta)

    def get_module(self, data):

        if isinstance(data, torch.Tensor):
            mod = torch
        else:
            mod = np
        return mod

    def transform(self, data, error=None):
        mod = self.get_module(data)
        if mod == torch:
            data_ = data
            error_ = error
        else:
            data_ = torch.from_numpy(data)
            error_ = torch.from_numpy(error)

        new_data = self.dist.cdf(data_)
        if mod == np:
            new_data = new_data.numpy()
        if error is not None:
            #dx = self.dist.log_prob(data_)
            #new_error = mod.exp(mod.log(error_) + dx)
            new_error = self.dist.cdf(data_ - error_)
            if mod == np:
                new_error = new_error.numpy()
            return new_data, new_error
        return new_data

    def inverse_transform(self, data, error=None):

        mod = self.get_module(data)
        if mod == torch:
            data_ = data
            error_ = error
        else:
            data_ = torch.from_numpy(data)
            error_ = torch.from_numpy(error)

        new_data = self.dist.icdf(data_)
        if mod == np:
            new_data = new_data.numpy()
        if error is not None:
            #dx = -self.beta / (data_ * mod.log(data_))
            #new_error = mod.sqrt((error_ * dx)**2)
            new_error = new_data - self.dist.icdf(error_)
            if mod == np:
                new_error = new_error.numpy()
            return new_data, new_error
        
        return new_data




class SemilogTransform:

    def __init__(self):

        pass

    def get_module(self, data):

        if isinstance(data, torch.Tensor):
            mod = torch
        else:
            mod = np
        return mod

    def transform(self, data, error=None):
        mod = self.get_module(data)

        new_data = mod.sign(data) * mod.log10(1 + mod.abs(data))
        if error is not None:
            new_error = error / (mod.abs(data) + 1) / np.log(10)
            return new_data, new_error
        return new_data
    
    def inverse_transform(self, transformed_data, error=None):
        mod = self.get_module(transformed_data)

        new_data = mod.sign(transformed_data) * (10**(mod.abs(transformed_data)) - 1)
        if error is not None:
            new_error = error * (mod.abs(new_data) + 1) * np.log(10)
            return new_data, new_error
        return new_data
    
class MaxTransform:

    def __init__(self, max_value):

        self.max_value = max_value

    def transform(self, data, error=None):

        new_data = data / self.max_value - 0.5
        if error is not None:
            new_error = error / self.max_value
            return new_data, new_error
        return new_data
    
    def inverse_transform(self, data, error=None):

        new_data = (data + 0.5) * self.max_value
        if error is not None:
            new_error = error * self.max_value
            return new_data, new_error
        return new_data

class TransformPipeline:

    def __init__(self, *transforms):

        self.transforms = transforms

    def transform(self, data, error=None):

        for transform in self.transforms:
            output = transform.transform(data, error)
            if isinstance(output, tuple):
                data, error = output
            else:
                data = output
        return output
    
    def inverse_transform(self, data, error=None):

        for transform in self.transforms[::-1]:
            output = transform.inverse_transform(data, error)
            if isinstance(output, tuple):
                data, error = output
            else:
                data = output
        return output