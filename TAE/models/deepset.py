import torch
import torch.nn as nn
from TAE.util import register_module
from typing import Union
import torch.nn.functional as F
from TAE.models.ModelNN import ResNet
from TAE.models.embeddings import LinearTimeEmbedding, FPEmbedding

NUM_BANDS = 6  # keep the same constants you use elsewhere
BAND_EMB_DIM = 12


@register_module
class EquivariantDeepSetEncoder(nn.Module):

    def __init__(
        self,        
        input_dim: int = 3,
        hidden_dim: int = 128,
        pooling: str = "mean",
        dropout: float = 0.2,
        activation: str = "ReLU",
        BAND_EMB_DIM: int = 12,
        NUM_BANDS: int = 6,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_bands = NUM_BANDS
        self.band_emb_dim = BAND_EMB_DIM
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)
        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=True)

        self.to_hidden = nn.Linear(input_dim + BAND_EMB_DIM, hidden_dim - BAND_EMB_DIM)
        self.ds_layers = []
        for i in range(num_layers):
            rho_layer = ResNet(
                hidden_dim - BAND_EMB_DIM, 
                hidden_dim - BAND_EMB_DIM, 
                nlayers=4, 
                nhid=hidden_dim - BAND_EMB_DIM, 
                activation=activation, 
                dropout=dropout
            )
            self.ds_layers.append(
                nn.ModuleList([
                    DeepSetEncoder(hidden_dim - BAND_EMB_DIM, 
                        hidden_dim - BAND_EMB_DIM, 
                        pooling, 
                        dropout, 
                        activation, 
                        BAND_EMB_DIM, 
                        NUM_BANDS
                    ),
                    rho_layer
                ])

            )
        self.ds_layers = nn.ModuleList(self.ds_layers)
        self.out = DeepSetEncoder(hidden_dim - BAND_EMB_DIM, 
                hidden_dim, 
                pooling, 
                dropout, 
                activation, 
                BAND_EMB_DIM, 
                NUM_BANDS
                )

        

    def forward(self, x: torch.Tensor, band_idx: torch.Tensor, mask: Union[torch.Tensor, None] = None):

        e_b = self.band_emb(band_idx[:, :, 0])
        x = torch.cat([x, e_b], dim=-1)
        x = self.to_hidden(x) # (B, S, E)
        x[~mask] = 0.0
        for m, rho in self.ds_layers:
            x = rho(m(x, band_idx)).unsqueeze(1) + x # (B, S, E)
        x = self.out(x, band_idx)
        return x



@register_module
class SimpleDeepSet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        pooling: str = "sum",
        dropout: float = 0.2,
        activation: str = "PReLU",
        norm:bool = False,
    ):
        super().__init__()
        if pooling not in {"mean", "sum", "max", "all", "attentive"}:
            raise ValueError("pooling must be one of {'mean', 'sum', 'max', 'all', 'attentive'}")
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = getattr(nn, activation)
        self.phi_out = self.activation()
        self.phi = ResNet(input_dim, hidden_dim, nlayers=2, nhid=hidden_dim, activation=activation, dropout=dropout, norm=norm)
        self.rho = ResNet(hidden_dim, hidden_dim, nlayers=2, nhid=hidden_dim, activation=activation, dropout=dropout, norm=norm)
        self.pooling = pooling  # commutative aggregation
        if self.pooling == "all":
            self.transform_shapes = nn.Linear(hidden_dim*4, hidden_dim)
        elif self.pooling == "attentive":
            self.W = nn.Linear(hidden_dim, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            self.tanh = nn.Tanh()

        #self.out_final = nn.Sequential(
        #    LearnedMLPBlock(hidden_dim, hidden_dim, hidden_dim=None, activation=activation),
        #)


    # --- helper ------------------------------------------------------------
    def _aggregate(self, h: torch.Tensor, mask=None):
        """
        h    : (B, N, hidden_dim)
        mask : (B, N)  with 1 for valid points, 0 for padding
        """
        if self.pooling == "mean":
            return h.mean(dim=1)
        elif self.pooling == "sum":
            #return h.sum(dim=1)
            return h.sum(dim=1)
        elif self.pooling == "max":
            return torch.max(h, dim=1)[0] #.max(dim=1).values
        elif self.pooling == "min":
            return torch.min(h, dim=1)[0]
        elif self.pooling == "all":
            return self.transform_shapes(torch.cat([h.mean(dim=1), h.sum(dim=1), torch.max(h, dim=1)[0], torch.min(h, dim=1)[0]], dim=1))
        elif self.pooling == "attentive":
            score = self.v(self.tanh(self.W(h))) # (B, S, 1)
            weights = F.softmax(score, dim=1) # (B, S, 1)
            pooling = (weights * h).sum(dim=1) # (B, E)
            return pooling

        # mean (default) – optionally masked    
        masked = h * mask.unsqueeze(-1)
        total  = masked.sum(dim=1)
        count  = mask.sum(dim=1, keepdim=True)#.clamp_min(1)
        return total / count

    # --- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None):
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

        h = self.phi_out(self.phi(x))                  # element-wise encoding (phi)
        h = self._aggregate(h, mask)     # permutation-invariant pooling (Sigma)
        z = self.rho(h)                  # set-level mapping (rho)
        z = h
        return z

@register_module
class DeepSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        pooling: str = "sum",
        dropout: float = 0.2,
        activation: str = "ReLU",
        BAND_EMB_DIM: int = 12,
        NUM_BANDS: int = 6,
    ):
        super().__init__()
        if pooling not in {"mean", "sum", "max", "min", "all"}:
            raise ValueError("pooling must be one of {'mean', 'sum', 'max', 'min', 'all'}")
        
        self.num_bands = NUM_BANDS
        self.band_emb_dim = BAND_EMB_DIM
        self.hidden_dim = hidden_dim
        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=True)
        self.dropout = dropout
        self.activation = getattr(nn, activation)
        self.time_embedding = LinearTimeEmbedding(emb_dim=input_dim)
        self.float_embedding = FPEmbedding(1, input_dim)
        self.phi_out = self.activation()
        #self.phi_out = BinaryGELU()

        # phi network – shared across points
        self.phi = ResNet(input_dim + BAND_EMB_DIM, hidden_dim, nlayers=4, nhid=hidden_dim, activation=activation, dropout=dropout)
        #self.phi = nn.Sequential(    
        #    nn.Linear(input_dim + BAND_EMB_DIM, hidden_dim),
        #    self.activation(),
        #    nn.Dropout(dropout),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    self.activation(),
        #    nn.Dropout(dropout),
        #)

        self.pooling = pooling  # commutative aggregation
        if self.pooling == "all":
            self.transform_shapes = nn.Linear(hidden_dim*4, hidden_dim)
        #self.softmax = nn.LogSoftmax(dim=-1)

        
        #self.norm = nn.BatchNorm1d(input_dim)

        # rho network – maps set embedding to latent code
        #self.rho = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    self.activation(),
        #    nn.Linear(hidden_dim, hidden_dim),
        #)

    # --- helper ------------------------------------------------------------
    def _aggregate(self, h: torch.Tensor, mask: Union[torch.Tensor, None] = None):
        """
        h    : (B, N, hidden_dim)
        mask : (B, N)  with 1 for valid points, 0 for padding
        """
        if mask is None:
            return h.mean(dim=1)
        elif self.pooling == "sum":
            #return h.sum(dim=1)
            return (h*mask.unsqueeze(-1)).sum(dim=1)
        elif self.pooling == "mean":
            return h * mask.unsqueeze(-1).sum(dim=1)/mask.sum(dim=1, keepdim=True)

        elif self.pooling == "max":
            return torch.max(h * mask.unsqueeze(-1), dim=1)[0] #.max(dim=1).values

        elif self.pooling == "min":
            return torch.min(h * mask.unsqueeze(-1), dim=1)[0]

        elif self.pooling == "all":
            return self.transform_shapes(
                torch.cat(
                    [
                        h * mask.unsqueeze(-1).sum(dim=1)/mask.sum(dim=1, keepdim=True), 
                        (h*mask.unsqueeze(-1)).sum(dim=1), 
                        torch.max(h * mask.unsqueeze(-1), dim=1)[0], 
                        torch.min(h * mask.unsqueeze(-1), dim=1)[0]
                        ], dim=1
                    )
                )


        # mean (default) – optionally masked
        
        

        masked = h * mask.unsqueeze(-1)
        total  = masked.sum(dim=1)
        count  = mask.sum(dim=1, keepdim=True)#.clamp_min(1)
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
        t = x[..., :1]
        t_emb = self.time_embedding(t)
        f = x[..., 1:2]
        f_emb = self.float_embedding(f)
        x = f_emb + t_emb
        # x = torch.transpose(self.norm(torch.transpose(x, 1, 2)), 1, 2)
        x = torch.cat([x, e_b], dim=-1)
        h = self.phi_out(self.phi(x))                  # element-wise encoding (phi)
        h = self._aggregate(h, mask)     # permutation-invariant pooling (Sigma)
        #z = self.rho(h)                  # set-level mapping (rho)
        z = h
        return z
