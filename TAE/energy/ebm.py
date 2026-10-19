import torch
from torch import nn
import torch.functional as F

from TAE.util import register_module
from TAE.energy import sampler



@register_module
class EnergyBasedModel(nn.Module):

    def __init__(self, model, sampler):

        super().__init__()

        self.model = model
        self.sampler = sampler
        self.sampler.model = model


    def forward(self, x):

        return self.model(x)

    def sample_new_exmps(self, *args, **kwargs):

        return self.sampler.sample_new_exmps(*args, **kwargs)

    def generate_samples(*args, **kwargs):

        return self.sampler.generate_samples(*args, **kwargs)