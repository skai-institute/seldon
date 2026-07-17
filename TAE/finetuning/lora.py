import numpy as np
import torch
from torch import nn
from functools import reduce
from TAE.util import register_module
import copy
import sys

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.0)

def init_zeros(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.0)

@register_module
class LoRA(nn.Module):
    '''
    https://arxiv.org/pdf/2106.09685
    '''

    def __init__(self, original_module, input_dim, hidden_dim, output_dim, alpha=1.0, conv=False):

        super().__init__()

        self.conv = conv
        self.original_module = copy.deepcopy(original_module)
        if self.conv:
            self.l1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False, groups=self.original_module.groups)
            self.l2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, bias=False, groups=self.original_module.groups)
        else:
            self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
            self.l2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.lora = nn.Sequential(self.l1, # A
                                  self.l2 # B
                                 )
        self.alpha = alpha
        self.r = hidden_dim
        self.scale = self.alpha / self.r
        init_weights(self.l1)
        init_zeros(self.l2)
        
    def forward(self, x):

        return self.original_module(x) + self.lora(x) * self.scale

    def extra_repr(self) -> str:

        return f'''original_module={self.original_module}, hidden_dim={self.r}, scale={self.scale}={self.alpha}/{self.r}'''

    def get_fused_weights(self):

        if self.conv:

            return self.original_module.weight.data + self.scale * (self.l2.weight.data.squeeze(-1).repeat(1, self.original_module.groups) @ self.l1.weight.data.squeeze(-1)).unsqueeze(-1)

        return self.original_module.weight.data + self.scale * (self.l2.weight.data @ self.l1.weight.data)



@register_module
class LoRATuner(nn.Module):
    '''
    A class for managing LoRA (Low-Rank Adaptation) to a model's linear layers
    '''

    def __init__(self, model, low_rank_size=None, low_rank_scale=None, from_pretrained=True, freeze_bias=False, alpha=1.0, square_matrix_only=False, min_hidden_dim=1, hit_convs=False):

        super().__init__()

        assert low_rank_scale is not None or low_rank_size is not None, "low_rank_scale or low_rank_size must be provided"
        assert low_rank_scale is None or low_rank_size is None, "low_rank_scale and low_rank_size cannot both be provided"

        self.model = model
        self.square_matrix_only = square_matrix_only
        self.from_pretrained = from_pretrained
        self.freeze_bias = freeze_bias
        self.alpha = alpha
        self.min_hidden_dim = min_hidden_dim
        self.hit_convs = hit_convs # Apply tuning to the 1DConvolutions used in multihead layers

        if low_rank_scale is not None:
            self.low_rank_scale = low_rank_scale
            self.mode = 'scale'

        elif low_rank_size is not None:
            self.low_rank_size = low_rank_size
            self.mode = 'constant'

        self.adapted = False

        self.linear_cache = nn.ModuleDict()
        self.lora_cache = nn.ModuleDict()
        #self.adapt_model()

    def train(self, mode=True):

        if mode:
            self.adapt_model()
        else:
            self.fuse_model()
        super().train(mode)


    def forward(self, *args, **kwargs):

        return self.model(*args, **kwargs)

    def __getattr__(self, attribute):

        try:
            return super().__getattr__(attribute)
        except:
            return getattr(super().__getattr__('model'),attribute)


    def __setattr__(self, attribute, value):

        try:
            super().__setattr__(attribute, value)
        except:
            setattr(super().__getattr__('model'), attribute, value)

    def get_lora_kwargs(self, module):

        if isinstance(module, nn.Linear):

            input_dim = module.in_features
            output_dim = module.out_features
            if self.square_matrix_only:
                if input_dim != output_dim:
                    return None
            hidden_dim = self.low_rank_size if self.mode == 'constant' else (int(self.low_rank_scale * min(input_dim, output_dim)) if (input_dim + output_dim) > 2 else 0)
            if hidden_dim < self.min_hidden_dim:
                return None

            return {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': hidden_dim, 'alpha':self.alpha, 'conv': False}

        elif self.hit_convs and isinstance(module, nn.Conv1d):
            input_dim = module.in_channels
            output_dim = module.out_channels
            if self.square_matrix_only:
                if input_dim != output_dim:
                    return None
            hidden_dim = self.low_rank_size if self.mode == 'constant' else  (int(self.low_rank_scale * min(input_dim, output_dim)) if (input_dim + output_dim) > 2 else 0)
            if hidden_dim < self.min_hidden_dim:
                return None
            hidden_dim = hidden_dim * module.groups

            return {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': hidden_dim, 'alpha':self.alpha, 'conv': True}

        else:
            return None


    def adapt_model(self):

        if self.adapted:
            return self.model
        print("LoRATuner: Adapting Model...", file=sys.stderr)

        if self.from_pretrained:
            for param in self.parameters():
                param.requires_grad = False

        substitutions = {}
        for name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, nn.Linear) or (self.hit_convs and isinstance(module, nn.Conv1d)):
                if '.l1' not in name and '.l2' not in name:
                    # Be careful to infinitely add lora parameters
                    if name.replace('.', '__') not in self.lora_cache:
                        lora_kwargs = self.get_lora_kwargs(module)
                        if lora_kwargs is None:
                            continue
                        if module not in self.linear_cache.values():
                            self.linear_cache[name.replace('.', '__')] = module # Just use the original
                            self.lora_cache[name.replace('.', '__')] = LoRA(module, **lora_kwargs).to(module.weight.device)
                            print("LoRATuner: Created LoRA Layer for", name, file=sys.stderr)
                        else:
                            for cached_name, cached_module in self.linear_cache.items():
                                if module is cached_module:
                                    self.lora_cache[name.replace('.','__')] = self.lora_cache[cached_name]
                                    self.linear_cache[name.replace('.', '__')] = self.linear_cache[cached_name]
                                    print("LoRATuner: Copied LoRA Layer for", name, "from", cached_name, file=sys.stderr)
                                    break
                        
                    substitutions[name] = self.lora_cache[name.replace('.', '__')]
        for name, module in substitutions.items():
            if module.original_module.bias is not None:
                module.original_module.bias.requires_grad = not self.freeze_bias
            for param in module.lora.parameters():
                param.requires_grad=True
            self.model.set_submodule(name, module)
            
        self.adapted = True

        return self.model

    def fuse_model(self):

        if not self.adapted:
            return self.model
        print("LoRATuner: Fusing Model...", file=sys.stderr)

        substitutions = {}

        for name, module in self.model.named_modules(remove_duplicate=False):
            if isinstance(module, LoRA):
                new_weight = module.get_fused_weights()
                new_linear = self.linear_cache[name.replace('.', '__')]

                new_linear.weight.data = new_weight
                if new_linear.bias is not None:
                    new_linear.bias.data = module.original_module.bias.data
                substitutions[name] = self.linear_cache[name.replace('.', '__')]
        for name, module in substitutions.items():
            self.model.set_submodule(name, module)

        self.adapted = False

        return self.model

        