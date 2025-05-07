import typing
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass

from .encoders import GRUEncoder
from .decoders import MLPBandDecoder 




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
        
        return {'loss': train_loss}




class BandAwareAutoencoder(nn.Module):
    """
    GRU-VAE-style autoencoder for supernova light curve reconstruction.
    (Single-band version, no band_idx used anymore.)

    This model uses a GRU encoder and an MLP decoder.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = MLPBandDecoder(latent_dim)  # hidden_dim inside decoder is default (128)

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
        z, z_mean, z_logvar = self.encoder(x, band_idx_part)
        reconstructed = self.decoder(z, t, band_idx_full)

        return {
            "reconstructed": reconstructed,
            "z_mean": z_mean,
            "z_logvar": z_logvar
        }

    def encode(self, x, band_idx):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            z, _, _ = self.encoder(x, band_idx)
            return z

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)
