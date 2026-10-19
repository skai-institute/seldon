# loss.py
from torch import nn
import torch
import numpy as np
from TAE.util import register_module


@register_module
class MaskedGaussianNLL(nn.Module):

    def __init__(self, 
                 eps: float = 1e-6,
                 weight_bands=True, 
                 weight_classes=True, 
                 inverse_transform=True,
                 ):

        super().__init__()
        self.eps = eps
        self.weight_bands = weight_bands
        self.weight_classes = weight_classes
        self.inverse_transform = inverse_transform

    def forward(self, out, batch, *args):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]
        class_weight = batch["class_weight"]
        band_weight = batch["band_weight"]

        if self.inverse_transform:
            scale = batch['flux_norm_info'][0][-1]
            flux_transform = lambda x: torch.sign((x+0.5)*scale) * (10 ** ((torch.absolute((x + 0.5)*scale).clamp_max(6.0))) - 1)

            target = flux_transform(
                target
            )
            sigma = (
                sigma
                * np.log(10.0)
                * (target + 1)
            )

            recon = flux_transform(
                recon
            )

        loss = (recon - target) ** 2 / sigma.clamp_min(self.eps).square()

        if self.weight_classes:
            loss = loss * class_weight[:, None, None] 
        if self.weight_bands:
            loss = loss * band_weight[:, :, None]

        loss = 0.5 * (pad_mask * loss).sum(1) / (pad_mask.sum(1) + self.eps)

        return loss

@register_module
class MaskedMSELoss(nn.Module):

    def __init__(self, 
                 eps: float = 1e-6,
                 weight_bands=True, 
                 weight_classes=True, 
                 inverse_transform=True,
                 ):
        super().__init__()
        self.eps = eps
        self.weight_bands = weight_bands
        self.weight_classes = weight_classes
        self.inverse_transform = inverse_transform

    def forward(self, out, batch, *args):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)
        class_weight = batch["class_weight"]
        band_weight = batch["band_weight"]


        if self.inverse_transform:
            scale = batch['flux_norm_info'][0][-1]
            flux_transform = lambda x: torch.sign((x + 0.5)*scale) * (10 ** ((torch.absolute((x + 0.5)*scale).clamp_max(6.0))) - 1)
            target = flux_transform(
                target
            )
            recon = flux_transform(
                recon
            )

        loss = (recon - target) ** 2

        if self.weight_classes:
            loss = loss * class_weight[:, None, None] 
        if self.weight_bands:
            loss = loss * band_weight[:, :, None]

        loss = (pad_mask * loss).sum(1) / (pad_mask.sum(1) + self.eps)

        return loss
    
@register_module
class FlexibleGaussianLoss:

    def __init__(self, 
                 eps: float = 1e-6, 
                 ):
        super().__init__()
        self.loss = nn.GaussianNLLLoss(full=True, reduction='none', eps=eps)
        self.eps = eps

    def __call__(self, input, target, sigma, *args):

        var = torch.square(sigma)
        return self.loss(input, target, var)

@register_module
class FlexibleHuberLoss(nn.Module):

    def __init__(self, 
                 eps: float = 1e-6, 
                 delta: float = 1.0, 
                 ):
        super().__init__()
        self.eps = eps
        self.delta = delta

    def forward(self, input, target, sigma, delta_scaler=None):

        delta = self.delta

        if delta_scaler is not None:
            delta_scale = delta_scaler.unsqueeze(-1) > 0
            delta = delta * delta_scale
            delta = delta.clamp_min(self.delta / 2)
            total_values = len(delta_scale.ravel())
            delta_weight = total_values / (delta_scale.sum() + 1) / 2 * delta_scale
            delta_weight += total_values / ((~delta_scale).sum() + 1) / 2 * ~delta_scale

        a = torch.abs((input - target) / sigma.clamp_min(self.eps))
        
        huber_mask = a < delta
        loss = delta * (a - 0.5 * delta)
        loss[huber_mask] = 0.5 * a[huber_mask].square()
        loss = loss * delta_weight

        return loss

@register_module
class MaskedHuberLoss(nn.Module):

    def __init__(self, 
                 eps: float = 1e-6, 
                 delta: float = 1.0, 
                 weight_bands=True, 
                 weight_classes=True, 
                 inverse_transform=True,
                 alpha: float = 0.9,
                 scale_delta_by = None,
                 scale_recon = True,
                 tune_parameters = False):
        super().__init__()
        self.eps = eps
        self.delta = delta
        self.weight_bands = weight_bands
        self.weight_classes = weight_classes
        self.inverse_transform = inverse_transform
        self.scale_recon = scale_recon
        self.tune_parameters = tune_parameters

        self.weighting_func = nn.Softmax(dim=1)
        self.weighting_func_out = nn.Softmin(dim=1)
        self.alpha = alpha 
        self.scale_delta_by = scale_delta_by
        assert scale_delta_by is None or isinstance(scale_delta_by, str), f"scale_delta_by must be str or None but got {scale_delta_by}"

    def forward(self, out, batch):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]
        time = batch["time_full"]
        in_sample_mask = batch["in_sample_mask"].unsqueeze(-1)
        in_sample_mask = in_sample_mask & pad_mask
        out_sample_mask = ~in_sample_mask & pad_mask
        class_weight = batch["class_weight"]
        band_weight = batch["band_weight"]
        #weights = (
        #    self.weighting_func(time + torch.log(in_sample_mask)) * self.alpha
        #    + (1 - self.alpha) / (in_sample_mask.sum(1).unsqueeze(1) + self.eps)
        #) * in_sample_mask.sum(1).unsqueeze(1)
        #weights[out_sample_mask] = (
        #    (
        #        self.weighting_func_out(time - torch.log(out_sample_mask)) * self.alpha
        #        + (1 - self.alpha) / (out_sample_mask.sum(1).unsqueeze(1) + self.eps)
        #    )
        #    * out_sample_mask.sum(1).unsqueeze(1)
        #)[out_sample_mask]
        #weights[in_sample_mask] = 0.0

        #weights = (
        #    (
        #        self.weighting_func_out(time - torch.log(pad_mask)) * self.alpha
        #        + (1 - self.alpha) / (pad_mask.sum(1).unsqueeze(1) + self.eps)
        #    )
        #    * pad_mask.sum(1).unsqueeze(1)
        #)

        
        if self.inverse_transform:# 1000:
            scale = batch['flux_norm_info'][0][-1]
            flux_transform = lambda x, scale: torch.sign((x+0.5)*scale) * (10 ** ((torch.absolute((x + 0.5)*scale).clamp_max(9.0)-3))*1000 - 1)

            target = flux_transform(
                target, scale
            )
            sigma = (
                sigma
                * scale
                * np.log(10.0)
                * (torch.absolute(target) + 1)
            )
            if not self.scale_recon: # Don't apply the min-max scaling to the reconstruction (assume it is already applied)
                scale = 1.0 

            recon = flux_transform(
                recon, scale
            )
        
        delta = self.delta
        if self.scale_delta_by is not None:
            scaling = batch[self.scale_delta_by].unsqueeze(-1) > 0
            delta = delta * scaling
            delta = delta.clamp_min(self.delta / 2)
            total_values = len(scaling.ravel())
            delta_weight = total_values / (scaling.sum() + 1) / 2 * scaling
            delta_weight += total_values / ((~scaling).sum() + 1) / 2 * ~scaling

        a = torch.abs((recon - target) / sigma.clamp_min(self.eps))
        
        huber_mask = a < delta
        loss = delta * (a - 0.5 * delta)
        loss[huber_mask] = 0.5 * a[huber_mask].square()
        loss = loss

        if self.weight_classes:
            loss = loss * class_weight[:, None, None] 
        if self.weight_bands:
            loss = loss * band_weight[:, :, None]
        if self.scale_delta_by is not None:
            loss = loss * delta_weight # weight the loss by fraction of points

        if self.tune_parameters:
            parameters = out['parameters']
            if 'mu' in parameters:
                mu = parameters['mu'] # (B, T, E, K) or (B, T, E, N, K)
            else:
                mu = parameters['offset']

            if 'weights' in parameters:
                weights = parameters['weights']
            else:
                weights = parameters['amplitude']
            b, t, e, k = mu.shape
            mu_mean = mu.mean(dim=-1, keepdim=True) # center in basis-space
            mu_diff = (mu - mu_mean)
            #mu_loss = (mu_diff**2).mean(3).mean(2)[:, :, None]
            weight_loss = torch.abs(weights)
            param_loss = torch.mean(weight_loss * mu_diff**2, dim=(-1, -2))[:, :, None]
            #import sys
            #print(mu.shape, pad_mask.shape, file=sys.stderr)
            #mu_loss = (mu**2).mean((2,3))[:, :, None]
            loss += param_loss * 1e-2

        #assert (loss >= 0).all(), f"{delta.min(), delta.max(), delta_weight.min(), delta_weight.max(), scaling.min(), scaling.max(), loss.min(), loss.max()}"


        loss = (pad_mask * loss).sum(1) / (pad_mask.sum(1) + self.eps) 

        return loss
