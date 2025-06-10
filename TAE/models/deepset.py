import torch
import torch.nn as nn
from TAE.util import register_module
from typing import Union
import torch.nn.functional as F

NUM_BANDS = 6  # keep the same constants you use elsewhere
BAND_EMB_DIM = 6

@register_module
class DeepSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        pooling: str = "mean",
        dropout: float = 0.2,
        activation: str = "Softplus",
    ):
        super().__init__()
        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of {'mean', 'sum', 'max'}")
        
        self.hidden_dim = hidden_dim
        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM)
        self.dropout=dropout
        self.activation = getattr(nn, activation)

        # phi network – shared across points
        self.phi = nn.Sequential(    
            nn.Linear(input_dim + BAND_EMB_DIM, hidden_dim),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Dropout(dropout),
        )

        self.pooling = pooling  # commutative aggregation

        # rho network – maps set embedding to latent code
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # --- helper ------------------------------------------------------------
    def _aggregate(self, h: torch.Tensor, mask: Union[torch.Tensor, None] = None):
        """
        h    : (B, N, hidden_dim)
        mask : (B, N)  with 1 for valid points, 0 for padding
        """
        if self.pooling == "sum":
            return h.sum(dim=1)

        elif self.pooling == "max":
            return h.max(dim=1).values

        # mean (default) – optionally masked
        elif mask is None:
            return h.mean(dim=1)
        

        masked = h * mask.unsqueeze(-1)
        total  = masked.sum(dim=1)
        count  = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return total / count

    # --- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, band_idx: torch.Tensor, mask: Union[torch.Tensor, None] = None):
        """
        Parameters
        ----------
        x    : (B, N, in_dim)
        band_idx: (B, N, 1)
        mask : (B, N)  optional boolean / 0-1 mask

        Returns
        -------
        z    : (B, out_dim)
        """

        e_b = self.band_emb(band_idx[:, :, 0])
        x = torch.cat([x, e_b], dim=-1)
        h = self.phi(x)                  # element-wise encoding (phi)
        h = self._aggregate(h, mask)     # permutation-invariant pooling (Sigma)
        z = self.rho(h)                  # set-level mapping (rho)
        return z