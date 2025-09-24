'''Module for handling masking operations from index augmenters and so forth'''
import numpy as np

class Masking:

    def __init__(self, length):

        self.length = length

    def transform(self, data, *args):

        pass


class PaddingMask(Masking):

    def transform(self, data, *args):

        data_length = len(data)
        
        padding_mask = np.zeros(self.length, dtype=bool)
        padding_mask[:data_length] = True

        return padding_mask
    
class AugmentationIndexMask(Masking):

    def transform(self, data, augmentation_index, *args):

        augmentation_mask = np.zeros(self.length, dtype=bool)
        augmentation_mask[augmentation_index] = True

        return augmentation_mask