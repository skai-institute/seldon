import typing
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass

from .encoders import GRUEncoder
from .decoders import MLPBandDecoder
from .parametric_decoders import BazinDecoder


@dataclass
class AEOutput:
    observation: torch.Tensor
    reconstruction: torch.Tensor
    latent_observation: torch.Tensor
    mask: torch.Tensor


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):  # Reduce latent space to 4
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),  # Replacing ReLU with LeakyReLU
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, input_dim),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, mask):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return AEOutput(
            observation=x,
            mask=mask,
            reconstruction=reconstructed,
            latent_observation=latent,
        )

    def loss_function(self, result):

        train_loss = self.criterion(
            result.reconstruction * result.mask,
            result.observation * result.mask,
        )

        return {"loss": train_loss}


class BandAwareAutoencoder(nn.Module):
    """
    GRU-VAE-style autoencoder for supernova light curve reconstruction.
    (Single-band version, no band_idx used anymore.)

    This model uses a GRU encoder and an MLP decoder.
    """

    def __init__(
        self,
        latent_dim=8,
        kl_weight=0.001,
        out_sample_lambda=0.8,
        smooth_weight=1.0,
        variational=True,
        encoder="GRUEncoder",
        decoder="MLP",
        loss='mse',
        **kwargs
    ):
        
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()
        self.encoder = encoder
        #self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim)
        # self.decoder = MLPBandDecoder(latent_dim, )  # hidden_dim inside decoder is default (128)

        self.decoder = decoder
        self.variational = variational

        self.fc_mean = nn.Linear(encoder.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder.hidden_dim, latent_dim)

        #if decoder == "MLP":
        #    self.decoder = MLPBandDecoder(latent_dim, **kwargs)
        #elif decoder == "Bazin":
        #    self.decoder = BazinDecoder(latent_dim, **kwargs)
        #else:
        #    raise NameError("Decoder must be MLP or Bazin.")

        self.kl_weight = kl_weight
        self.out_sample_lambda = out_sample_lambda
        self.smooth_weight = smooth_weight

        self.loss_method = loss

    def forward(self, x, t, band_idx_part, band_idx_full):
        """
        Forward pass through encoder and decoder.

        Args:
            x: (B, T1, input_dim) — input partial curve (time, flux, 1)
            t: (B, T2, 1) — full time grid for reconstruction
            band_idx_full: (B, T2, 1) — band code for each point

        Returns:
            dict with 'reconstructed', 'z_mean', 'z_logvar'
        """
        #z, z_mean, z_logvar = self.encoder(x, band_idx_part)
        memory = self.encoder(x, band_idx_part)
        z_mean = self.fc_mean(memory)
        z_logvar = torch.clamp(self.fc_logvar(memory), min=-10.0, max=2.0)

        if self.variational:
            if self.training:
                std = torch.exp(0.5 * z_logvar)
                eps = torch.randn_like(std)
                z = z_mean + eps * std
            else:
                z = z_mean
        else:
            z = z_mean
        reconstructed = self.decoder(z, t, band_idx_full)

        return {"reconstructed": reconstructed, "z_mean": z_mean, "z_logvar": z_logvar, 'z_sample': z}

    def encode(self, x, band_idx):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            z = self.encoder(x, band_idx)
            z = self.fc_mean(z)
            return z

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)

    #def loss_function(self, recon, target, time, mask, z_mean, z_logvar):
    def loss_function(self, out, batch):
        """
        Compute total loss = in-sample MSE + (out-of-sample MSE) + KL divergence + smoothness penalty.
        """
        recon=out["reconstructed"]
        target=batch["flux_full"]
        time=batch["time_full"]
        mask=batch["in_sample_mask"]
        z_mean=out["z_mean"]
        z_logvar=out["z_logvar"]
        sigma = batch["flux_err_full"]

        squared_error = (recon - target) ** 2
        if self.loss_method == 'NLL':
            squared_error /= sigma.square().clamp_min(1e-3)
        mask = mask.unsqueeze(-1)

        in_loss = (squared_error * mask).sum()
        out_loss = (
            (squared_error * (~mask)).sum()
            * ((~mask).sum() + 1e-6)
            / (mask.sum() + 1e-6)
        )
        if self.variational:
            kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        else:
            kl = 0

        total_loss = (
            in_loss + self.out_sample_lambda * out_loss + self.kl_weight * kl
        )  # +
        #   self.smooth_weight * smooth_penalty
        # )

        return {"loss": total_loss, 'squared_error': squared_error.sum(1)}

    def training_step(self, batch, batch_idx):
        out = self(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"],
        )

        loss = self.loss_function(out, batch)

        return loss
