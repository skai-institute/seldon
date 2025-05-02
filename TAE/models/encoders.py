import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BaseEncoder(nn.Module):
    """Base encoder specifying minimal init protocol."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: typing.Tuple[int, ...] = (32, 256),
        activation: str = "LeakyReLU",
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron encoder.

        Args:
            observation_dim: input feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(observation_dim=observation_dim)
        self.hidden_dims = hidden_dims

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = observation_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation_fn,
                    nn.Dropout(p=dropout),
                )
            )
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class Label_MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        label_dim: int = 4,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron encoder.

        Args:
            observation_dim: input feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(observation_dim=observation_dim)
        hidden_dims = [hidden_dim] * hidden_layers
        self.hidden_dims = hidden_dims
        self.is_residual_on = is_residual_on

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = int(observation_dim + label_dim)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation_fn,
                    nn.Dropout(p=dropout),
                )
            )
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.is_residual_on:
            outputs = self.mlp[0](inputs)
            outputs_pre = outputs
            outputs_curr = self.mlp[1](outputs_pre)
            outputs = outputs_curr
            for t in range(2, len(self.mlp)):
                outputs = self.mlp[t](outputs_pre + outputs_curr)
                outputs_pre = outputs_curr
                outputs_curr = outputs
        else:
            outputs = self.mlp(inputs)
        return outputs


class ConvEncoder(BaseEncoder):
    """Multi-layer perceptron encoder."""

    def __init__(
        self,
        observation_dim,
        kernel_size: int = 8,
        stride: int = 4,
        hidden_layers: int = 4,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
        depth: float = 3,
    ) -> None:
        """Initializer for multi-layer perceptron encoder.

        Args:
            observation_dim: input feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(observation_dim=observation_dim)

        self.is_residual_on = is_residual_on

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = tuple(list(map(int, observation_dim)))
        in_channels = observation_dim[0]
        out_channels = depth
        h_in = observation_dim[1]
        h_out = h_in
        P = 0
        for i in range(hidden_layers):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        (
                            self.observation_dim[0]
                            if i + 1 == hidden_layers
                            else out_channels
                        ),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=P,
                    ),
                    activation_fn,
                    nn.BatchNorm2d(
                        self.observation_dim[0]
                        if i + 1 == hidden_layers
                        else out_channels
                    ),
                    nn.Dropout(p=dropout),
                )
            )
            h_out = (h_in + 2 * P - kernel_size) // stride + 1
            h_in = h_out
            in_channels = out_channels
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
        self.mlp = nn.Sequential(*modules)
        self.h_out = h_out

        hidden_dims = [self.h_out] * hidden_layers
        self.hidden_dims = hidden_dims
        # self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=P)
        # self.h_out = (h_in + 2 * P - kernel_size)//stride + 1
        self.flat_output = self.h_out * self.h_out * self.observation_dim[0]
        self.linear_output = nn.Sequential(
            nn.Linear(self.flat_output, self.flat_output),
            activation_fn,
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, *self.observation_dim)
        if self.is_residual_on:
            outputs = self.mlp[0](inputs)
            outputs_pre = outputs
            outputs_curr = self.mlp[1](outputs_pre)
            for t in range(2, len(self.mlp)):
                outputs = self.mlp[t](outputs_pre + outputs_curr)
                outputs_pre = outputs_curr
                outputs_curr = outputs
        else:
            outputs = self.mlp(inputs)

        # outputs = self.pooling(outputs)
        outputs = outputs.view(-1, self.flat_output)
        outputs = self.linear_output(outputs)

        return outputs


# ==========================================
# Encoder: GRUEncoder (single-band, no band_idx)
# ==========================================

class GRUEncoder(nn.Module):
    """
    GRU-based encoder for single-band supernova light curves.
    Encodes [time, normalized flux] sequences into latent vector z.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim: number of input features (e.g., time + normalized flux + dummy 1)
            hidden_dim: hidden size of GRU
            latent_dim: dimension of latent z vector
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) input sequence

        Returns:
            z: sampled latent vector (B, latent_dim)
            z_mean: mean of latent distribution (B, latent_dim)
            z_logvar: log-variance of latent distribution (B, latent_dim)
        """
        _, h_n = self.gru(x)  # h_n: (1, B, hidden_dim)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)

        z_mean = self.fc_mean(h_n)
        z_logvar = torch.clamp(self.fc_logvar(h_n), min=-10.0, max=2.0)

        if self.training:
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std
        else:
            z = z_mean

        return z, z_mean, z_logvar
