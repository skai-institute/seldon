# loss.py
from torch import nn
import torch



class MaskedGaussianNLL(nn.Module):


    def __init__(self, eps: float=1e-6):

        super().__init__()
        self.eps = eps


    def forward(self, out, batch):

        recon=out["reconstructed"]
        target=batch["flux_full"]
        pad_mask=batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]

        squared_error = (recon - target) ** 2 / sigma.square().clamp_min(self.eps)

        loss = 0.5*(pad_mask * squared_error).sum(1)/(pad_mask.sum(1)+self.eps)

        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self, eps: float=1e-6):

        super().__init__()
        self.eps = eps


    def forward(self, out, batch):

        recon=out["reconstructed"]
        target=batch["flux_full"]
        pad_mask=batch["pad_mask_full"].unsqueeze(-1)

        squared_error = (recon - target) ** 2

        loss = (pad_mask * squared_error).sum(1)/(pad_mask.sum(1)+self.eps)

        return loss

class MaskedHuberLoss(nn.Module):

    def __init__(self, eps: float=1e-6, delta: float=1.0):
        super().__init__()
        self.eps = eps
        self.delta = delta

    def forward(self, out, batch):

        recon=out["reconstructed"]
        target=batch["flux_full"]
        pad_mask=batch["pad_mask_full"].unsqueeze(-1)
        sigma = batch["flux_err_full"]

        a = torch.abs((recon - target)/sigma.clamp_min(self.eps))
        huber_mask = a < self.delta
        loss = self.delta * (a - 0.5*self.delta)
        loss[huber_mask] = 0.5*a[huber_mask].square()

        loss = (pad_mask * loss).sum(1) / (pad_mask.sum(1)+self.eps)

        return loss
        
        






