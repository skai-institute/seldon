# loss.py
from torch import nn
import torch


class MaskedGaussianNLL(nn.Module):

    def __init__(self, eps: float = 1e-6):

        super().__init__()
        self.eps = eps

    def forward(self, out, batch):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]

        squared_error = (recon - target) ** 2 / sigma.square().clamp_min(self.eps)

        loss = 0.5 * (pad_mask * squared_error).sum(1) / (pad_mask.sum(1) + self.eps)

        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-6):

        super().__init__()
        self.eps = eps

    def forward(self, out, batch):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)

        squared_error = (recon - target) ** 2

        loss = (pad_mask * squared_error).sum(1) / (pad_mask.sum(1) + self.eps)

        return loss


class MaskedHuberLoss(nn.Module):

    def __init__(self, eps: float = 1e-6, delta: float = 1.0, alpha: float = 0.9):
        super().__init__()
        self.eps = eps
        self.delta = delta
        self.weighting_func = nn.Softmax(dim=1)
        self.weighting_func_out = nn.Softmin(dim=1)
        self.alpha = alpha

    def forward(self, out, batch):

        recon = out["reconstructed"]
        target = batch["flux_full"]
        pad_mask = batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]
        time = batch["time_full"]
        in_sample_mask = batch["in_sample_mask"].unsqueeze(-1)
        in_sample_mask = in_sample_mask & pad_mask
        out_sample_mask = ~in_sample_mask & pad_mask
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

        a = torch.abs((recon - target) / sigma.clamp_min(self.eps))
        huber_mask = a < self.delta
        loss = self.delta * (a - 0.5 * self.delta)
        loss[huber_mask] = 0.5 * a[huber_mask].square()
        loss = loss #* weights

        loss = (pad_mask * loss).sum(1) / (pad_mask.sum(1) + self.eps)

        return loss
