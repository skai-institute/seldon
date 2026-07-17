import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TAE.util import register_module, read_transmission_filter, map_transmission_filters
from TAE.datasets.utils import BAND_IDX_MAP
from TAE.models.basis import FourierActivation
from scipy.interpolate import interp1d
import sys

register_module(nn.Embedding)

UNIT_TABLE_SI = {
    'microns': 1e-6,
    'angstroms': 1e-10,
    'cm':1e-2,
    None: 1,
    }

@register_module
def LearnableBandEmbedding(bandpass, **kwargs):
    '''Learnable Band Embedding from a TAE.datasets.bandpass.BandPass object'''

    return nn.Embedding(bandpass.num_bands, **kwargs)

@register_module
class BandPassEmbedding(nn.Module):
    '''Static Band Embedding from a TAE.datasets.bandpass.BandPass object'''

    UNIT_SCALE = {
        'microns': 1.0,
        'angstroms': 1e4,
        'cm': 1e-4,
        }

    def __init__(self, bandpass, lambda_min=None, lambda_max=None, lambda_unit='microns', logscale=False, embedding_dim=512, max_norm=False):

        super().__init__()

        self.bandpass = bandpass
        self.emb_dim = bandpass.num_bands
        self.logscale = logscale

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.embedding_dim = embedding_dim

        self.lambda_unit = lambda_unit
        self.lambda_scale = UNIT_TABLE_SI[self.lambda_unit] / UNIT_TABLE_SI[self.bandpass.unit]

        self.max_norm = max_norm

        self._register_embedding()

    def _interpolate(self):

        wave_min = self.lambda_min / self.lambda_scale if self.lambda_min is not None else None
        wave_max = self.lambda_max / self.lambda_scale if self.lambda_max is not None else None
        n_points = self.embedding_dim

        all_waves = np.hstack([value[0] for value in self.bandpass.index_to_filter.values()])
        if wave_min is None:
            wave_min = all_waves.min()
        if wave_max is None:
            wave_max = all_waves.max()
        if n_points is None:
            n_points = len(np.unique(all_waves))

        wave_grid = np.logspace(np.log10(wave_min), np.log10(wave_max), n_points)
        interpolators = {key: interp1d(*value, fill_value=0.0, bounds_error=False) for key, value in self.bandpass.index_to_filter.items()}
        filter_embeddings = {key: value(wave_grid) for key, value in interpolators.items()}

        return wave_grid, filter_embeddings

    def _register_embedding(self):

        lambda_grid, filters = self._interpolate()

        throughput = np.vstack(list(filters.values())).astype(np.float32)
        if self.max_norm:
            throughput = throughput / throughput.max(axis=1, keepdims=True)
        if self.logscale:
            throughput = torch.log(torch.tensor(throughput, requires_grad=False)).clamp_min(-44.0)
        else:
            throughput = torch.tensor(throughput, requires_grad=False)
        self.emb = nn.Embedding.from_pretrained(throughput, freeze=True)
        lambda_grid = torch.tensor(lambda_grid[None, None, :].astype(np.float32), requires_grad=False) * self.lambda_scale
        self.num_bands = len(filters)
        self.register_buffer("lambda_grid", lambda_grid)

    def forward(self, band_idx):

        return self.emb(band_idx)


@register_module
class AttentiveBandPassEmbedding(BandPassEmbedding):

    def __init__(self, bandpass, lambda_min=None, lambda_max=None, lambda_unit='microns', logscale=False, embedding_dim=512, output_dim=256, activation='GELU', max_norm=False, **mht_kwargs):

        super().__init__(bandpass, lambda_min=lambda_min, lambda_max=lambda_max, lambda_unit=lambda_unit, logscale=logscale, embedding_dim=embedding_dim, max_norm=max_norm)
        assert bandpass.index_to_filter is not None, "AttentiveBandPassEmbedding currently only works with static transmission functions"
        if 'embed_dim' not in mht_kwargs:
            mht_kwargs.update({'embed_dim':output_dim})
        if 'num_heads' not in mht_kwargs:
            mht_kwargs.update({'num_heads': 1})
        mht_kwargs.update({'batch_first': True})

        activation = getattr(nn, activation)

        self.wave_embedding = nn.Sequential(nn.Linear(1, mht_kwargs['embed_dim']//2), FourierActivation(), nn.Linear(mht_kwargs['embed_dim'], mht_kwargs['embed_dim']))
        self.transmission_embedding = nn.Sequential(nn.Linear(1, mht_kwargs['embed_dim']), activation(), nn.Linear(mht_kwargs['embed_dim'], mht_kwargs['embed_dim']))
        self.layer_norm_transmission = nn.LayerNorm(mht_kwargs['embed_dim'])

        self.query = nn.Parameter(torch.randn((1, 1, mht_kwargs['embed_dim']), requires_grad=True))
        self.attention = nn.MultiheadAttention(**mht_kwargs)

        self.layer_norm_ff = nn.LayerNorm(mht_kwargs['embed_dim'])
        self.feed_forward = nn.Sequential(nn.Linear(mht_kwargs['embed_dim'], mht_kwargs['embed_dim']), activation(), nn.Linear(mht_kwargs['embed_dim'], mht_kwargs['embed_dim']), nn.LayerNorm(mht_kwargs['embed_dim']))
        self.output_projection = nn.Linear(mht_kwargs['embed_dim'], output_dim)
        

    def forward(self, band_idx):
        '''
             Query             Wavelength          Transmission
               |                   |                    |
               |                   V                    V
               |               Pos Encoding            MLP
               |                   |                    |
               |                   L------->(+)<--------⅃
               |                             |
               |                             V
               |                         LayerNorm
               |                             |
               |                             V
               |                         Key, Value
               |                             |
               L------>Cross Attention<------⅃
               |               |
               |               V
               |           Layer Norm
               |               |
               L----->(+)<-----⅃
               |       |
               |       V
               |  Feed Forward
               |       |
               |       V
               |   Layer Norm
               |       |
               L->(+)<-⅃
                   |
                   V
               Projection   
        '''

        B, S, *_ = band_idx.shape
        lambda_grid = self.lambda_grid.reshape(1, self.emb.embedding_dim, 1) # 1, 1, E -> 1, E, 1

        # This is a little trick that saves memory if we have fewer filter functions than we have data points.  Perform the op on all embeddings
        # Then select the ones we want at the end from the band index input
        sparse_transmission = self.transmission_embedding(self.emb(torch.arange(self.emb.num_embeddings, device=band_idx.device))[:, :, None]) # (N, E, emb_dim)
        transmission_emb = self.layer_norm_transmission(self.wave_embedding(lambda_grid) + sparse_transmission)  # N, E, embed_dim

        query = self.query.expand(transmission_emb.shape[0], 1, -1) # N, 1, embed_dim 

        pooled, _ = self.attention(query, transmission_emb, transmission_emb, need_weights=False) # N, 1, embed_dim
        feed_forward_input = self.layer_norm_ff(pooled) + query # N, 1, embed_dim
        output_embedding = self.output_projection(self.feed_forward(feed_forward_input) + query).squeeze(1)[band_idx.reshape(-1)].reshape(B, S, -1) # B, S, output_dim

        return output_embedding


@register_module
class TransmissionFilterEmbedding(nn.Module):

    UNIT_SCALE = {
        'microns': 1.0,
        'angstroms': 1e4,
        'cm': 1e-4,
        }

    def __init__(self, filelist, emb_dim=6, lambda_min=None, lambda_max=None, lambda_unit='microns', logscale=False):

        super().__init__()
        self.filelist = filelist # List of transmission filter files
        self.emb_dim = emb_dim
        self.logscale = logscale

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.lamba_unit = lambda_unit
        self.lambda_scale = self.UNIT_SCALE[self.lambda_unit]

        self.load_filters()

    def __len__(self):

        return self.num_bands


    def load_filters(self):

        wave_min = self.lambda_min / self.lambda_scale if self.lambda_min is not None else None
        wave_max = self.lambda_max / self.lambda_scale if self.lambda_max is not None else None
        lambda_grid, filters = map_transmission_filters(self.filelist, n_points=self.emb_dim, wave_min=wave_min, wave_max=wave_max)
        band_idx = [BAND_IDX_MAP[key] for key in filters]
        throughput = np.vstack(list(filters.values())).astype(np.float32)
        if self.logscale:
            throughput = torch.log(torch.tensor(throughput[np.argsort(band_idx)], requires_grad=False)).clamp_min(-44.0)
        self.emb = nn.Embedding.from_pretrained(throughput, freeze=True)
        lambda_grid = torch.tensor(lambda_grid[None, None, :].astype(np.float32), requires_grad=False) * self.lambda_scale
        self.num_bands = len(filters)
        self.register_buffer("lambda_grid", lambda_grid)

    def forward(self, band_idx):

        return self.emb(band_idx)



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
        pe = pe.unsqueeze(0)
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

        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)

@register_module
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
