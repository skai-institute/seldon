import typing
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass


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

