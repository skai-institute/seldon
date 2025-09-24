import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TAE.util import register_module

class LinearTimeEmbedding(nn.Module):
    # https://arxiv.org/html/2409.20092v1

    def __init__(self, emb_dim=1):

        super().__init__()

        self.emb_dim = emb_dim
        self.transform = nn.Linear(1, emb_dim) # mx+k

    def forward(self, t):

        return self.transform(t)
    
@register_module
class BatchEmbedding(nn.Module):

    def __init__(self, output_dim=None, **embedding_config): # Could make this just take kwargs
        super().__init__()

        self.embedding_keys = embedding_config.keys()
        self.embedders = embedding_config.values()
        self.output_dim = output_dim
        self.to_output = None

    def forward(self, batch):

        embeddings = [embedder(batch[key]) for key, embedder in zip(self.embedding_keys, self.embedders)]
        embeddings = torch.cat(embeddings, dim=-1)

        if self.to_output is None:
            if self.output_dim is not None:
                self.to_output = nn.Linear(embeddings.shape[-1], self.output_dim)
            else:
                return embeddings
        return self.to_output(embeddings)
    
class OneHotEmbedding(nn.Module):

    def __init__(self, num_classes=-1):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, x):

        return F.one_hot(x, num_classes=self.num_classes)

@register_module
class ConfigurableEmbedding(BatchEmbedding):

    def __init__(self, 
                 output_dim=None,
                 embed_bands=True,
                 embed_flux=True, 
                 embed_flux_errors=True,
                 embed_non_detections=True,
                 num_bands=6,
                 band_emb_dim=6,
                 flux_emb_dim=6,
                 flux_error_emb_dim=6,
                 non_detection_emb_dim=6,
                 ):
        
        embedders = [
            FPEmbedding(1, flux_emb_dim),
            FPEmbedding(1, flux_error_emb_dim),
            nn.Embedding(num_bands, band_emb_dim),
            nn.Embedding(2, non_detection_emb_dim)
            ]
        embedding_keys=["flux_full", "flux_error_full", "band_idx", "detection_flag"]

        embedding_config = dict(zip(embedding_keys, embedders))
        if not embed_bands:
            del embedding_config['band_idx']
        if not embed_flux:
            del embedding_config['flux_full']
        if not embed_flux_errors:
            del embedding_config['flux_error_full']
        if not embed_non_detections:
            del embedding_config['detection_flag']
        
        super().__init__(output_dim=output_dim, **embedding_config)




class PlotterEmbedding(nn.Module):
    '''Turn a time-series into a plot image with a learned manifold transformation'''


    def __init__(self, N_width=100, N_height=100, gamma=1.0, output_dim=64, device=None):
        super().__init__()

        self.N_width = N_width
        self.N_height = N_height
        self.gamma = gamma # Gaussian Width
        self.output_dim = output_dim

        self.time_sigmoid = nn.Sigmoid()
        self.flux_sigmoid = nn.Sigmoid()

        self.XX = None
        self.YY = None

        self.out = nn.Linear(self.N_height*6, self.output_dim)
        #self.squeeze = nn.Linear(6, 1)
        


    def forward(self, t, f, band_idx, mask=None):


        if self.XX is None:
            self.XX, self.YY = torch.meshgrid(
                torch.arange(t.shape[1], device=t.device, dtype=torch.float32), 
                torch.arange(self.N_height, device=t.device, dtype=torch.float32), 
                indexing='ij'
                ) # (h, w)

        t_map = (self.time_sigmoid(t)-0.5)*2*t.shape[1]# (B, S)
        f_map = self.flux_sigmoid(f)*self.N_height # (B, S)
        if mask is not None:
            f_map[~mask] = -10.0 * self.gamma
            t_map[~mask] = -10.0 * self.gamma
        XX = self.XX[None, :].expand(t.shape[0], -1, -1) # (B, w, h)
        YY = self.YY[None, :].expand(t.shape[0], -1, -1) # (B, w, h)
        grid = 1/(np.sqrt(2*np.pi*self.gamma))*torch.exp(-((YY-f_map[..., None])**2 + (XX-t_map[..., None])**2)/(2*self.gamma**2))
        grid = grid[..., None].expand(t.shape[0], -1, -1, 6) # (B, S, h, E)
        band_map = F.one_hot(band_idx, num_classes=6)[:, :, None, :].expand(*grid.shape)
        grid = grid.clone()*band_map
        #grid = self.squeeze(grid).squeeze(3) # (B, S, h)
        emb = self.out(grid.flatten(2)) #(B, S, E)
        return emb

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FPEmbedding(nn.Module):

    def __init__(self, nvector, nemb, dropout=0.2, normalize=False):
        super().__init__()

        self.nvector = nvector  # size of the input vector to embed
        self.nemb = nemb  # embedding dimension
        self.dropout = nn.Dropout(p=dropout)

        self.linear_transform = nn.Linear(nvector, nemb)
        self.norm = nn.LayerNorm(nemb)
        self.normalize = normalize

        self.init_weights()

    def init_weights(self):

        for layer in [self.linear_transform]:
            if isinstance(layer, nn.Linear):
                if hasattr(layer, "weight"):
                    nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # input is assumed to be N_seq x N_batch x N_vec
        if x.shape[-1] != self.nvector:
            if self.nvector != 1:
                raise ValueError("Input Vector Has the incorrect shape!")
            return self.forward(torch.unsqueeze(x, -1))
        x = self.linear_transform(x)
        if self.normalize:
            x = self.norm(x)
        return x
