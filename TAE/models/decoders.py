import typing

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from TAE.util import register_module


class BaseDecoder(nn.Module):
    """Base decoder specifying minimal init protocol."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs

@register_module
class MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: typing.Tuple[int, ...] = (64, 256),
        activation: str = "LeakyReLU",
    ) -> None:
        """Initializer for multi-layer perceptron decoder.

        Args:
            latent_dim: input latent space feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(latent_dim=latent_dim)
        self.hidden_dims = hidden_dims

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        dec = self.mlp(inputs)
        return dec


class Label_MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        output_dim: int = 1100,
        label_dim: int = 4,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron decoder.

        Args:
            latent_dim: input latent space feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(latent_dim=latent_dim)
        hidden_dims = [hidden_dim] * hidden_layers + [output_dim]
        self.hidden_dims = hidden_dims
        self.is_residual_on = is_residual_on

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = latent_dim + label_dim
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

        self.observation_dim = hidden_dims[-1]

    def std_to_bound(self, reconstruction_std):
        bound = (
            0.5
            * torch.prod(torch.tensor(self.observation_dim, dtype=torch.float32))
            * (
                1.0
                + torch.log(torch.tensor(2 * np.pi, dtype=torch.float32))
                + 2 * torch.log(torch.tensor(reconstruction_std, dtype=torch.float32))
            )
        )
        return bound

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_residual_on:
            outputs_pre = self.mlp[0](inputs)
            outputs_curr = self.mlp[1](outputs_pre)
            for t in range(2, len(self.mlp)):
                outputs = self.mlp[t](outputs_pre + outputs_curr)
                outputs_pre = outputs_curr
                outputs_curr = outputs
        else:
            outputs = self.mlp(inputs)
        return outputs


class ConvDecoder(BaseDecoder):
    """Multi-layer perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        kernel_size: int = 16,
        stride: int = 2,
        hidden_layers: int = 4,
        output_dim: int = 64,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
        depth: float = 3,
    ) -> None:
        """Initializer for multi-layer perceptron decoder.

        Args:
            latent_dim: input latent space feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(latent_dim=latent_dim)
        self.is_residual_on = is_residual_on
        self.latent_root = int(np.sqrt(self.latent_dim))

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = 1
        h_out = self.latent_root  # sqrt latent_dim
        P = ((stride - 1) * (output_dim - 1) - 1) // 2
        for h_dim in range(hidden_layers):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        3 if h_dim + 1 == hidden_layers else depth,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    activation_fn,
                    nn.BatchNorm2d(3 if h_dim + 1 == hidden_layers else depth),
                    nn.Dropout(p=dropout),
                )
            )
            in_channels = depth
            h_out = (h_out - 1) * stride + kernel_size
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            # in_channels = h_dim
        self.mlp = nn.Sequential(*modules)
        self.h_out = h_out
        self.observation_dim = output_dim * output_dim * 3
        self.output_dim = output_dim
        self.hidden_dims = [self.observation_dim]

        self.final_output = nn.Sequential(
            nn.Linear(self.h_out * self.h_out * 3, self.observation_dim), activation_fn
        )
        self.initial_input = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), activation_fn
        )

    def std_to_bound(self, reconstruction_std):
        bound = (
            0.5
            * torch.prod(torch.tensor(self.observation_dim, dtype=torch.float32))
            * (
                1.0
                + torch.log(torch.tensor(2 * np.pi, dtype=torch.float32))
                + 2 * torch.log(torch.tensor(reconstruction_std, dtype=torch.float32))
            )
        )
        return bound

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.initial_input(inputs)
        outputs = inputs.view(-1, 1, self.latent_root, self.latent_root)

        if self.is_residual_on:

            outputs = self.mlp[0](outputs)
            outputs_pre = outputs

            outputs_curr = self.mlp[1](outputs_pre)
            outputs = outputs_curr
            for t in range(2, len(self.mlp)):
                outputs = self.mlp[t](outputs_pre + outputs_curr)
                outputs_pre = outputs_curr
                outputs_curr = outputs

        else:
            outputs = self.mlp(outputs)
        outputs = outputs.view(-1, 3 * self.h_out * self.h_out)
        outputs = self.final_output(outputs)
        # outputs = outputs.view(-1, 3, self.output_dim, self.output_dim)
        return outputs


NUM_BANDS = 6  # keep the same constants you use elsewhere
BAND_EMB_DIM = 6
SIG_BASIS_K = 8  # number of sigmoid bases

@register_module
class MLPBandDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128, K: int = SIG_BASIS_K):
        super().__init__()
        self.K = K
        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM)

        inp = latent_dim + BAND_EMB_DIM  # concat z‖e_b
        self.mlp = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_BANDS * 3 * K),  # [a|c|s] for all bases
        )

        nn.init.uniform_(self.mlp[-1].weight, -0.2, 0.2)

    def forward(self, z, t, band_idx):  # t:(B,T,1)
        e_b = self.band_emb(band_idx[:, :, 0])  # (B,seq,E)
        z = z.unsqueeze(1).expand(z.shape[0], e_b.shape[1], z.shape[1])
        # import pdb; pdb.set_trace()
        params = self.mlp(
            torch.cat([z, e_b], dim=-1)
        )  # (B, 3K) # Should be (B, band, 3k)
        # params = torch.cat([mlp(z).unsqueeze(1) for mlp in self.mlps], dim=1) # perform with each mlp
        # params = self.mlp(z) # (B, band, 3K)
        # import pdb; pdb.set_trace()
        params = params.view(
            params.shape[0], params.shape[1], NUM_BANDS, 3 * self.K
        )  # (B,band,3K)
        params = (
            torch.gather(params, 2, band_idx.unsqueeze(2).expand(-1, -1, 1, 3 * self.K))
            .squeeze(2)
            .view(params.shape[0], -1, self.K, 3)
        )  # (B, K, 3)

        # params = torch.gather(params, 1, band_idx)
        # NEEDS ANOTHER DIMENSION TO INDEX EACH BAND!

        a = params[:, :, :, 0]  # (B,seq,K)
        c = params[:, :, :, 1]  # (B,seq,K)
        s = params[:, :, :, 2]  # (B,seq,K)
        # import pdb; pdb.set_trace()
        # ---------- broadcast & sum ----------------------------------
        t_exp = t.expand(-1, -1, self.K)  # (B,T,K)
        basis = torch.sigmoid((t_exp - c) * s)  # (B,T,K)
        out = (a * basis).sum(-1, keepdim=True)  # (B,T,1)
        return out
