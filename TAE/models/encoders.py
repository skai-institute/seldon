import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from TAE.util import register_module, map_transmission_filters
from TAE.datasets.utils import BAND_IDX_MAP
from TAE.models.embeddings import LinearTimeEmbedding, FPEmbedding, PlotterEmbedding, PositionalEncoding
from TAE.models.ModelNN import ResNet, Partial
from TAE.models.neuralode import RNNODEEncoder, ODEDecoder
from TAE.models.deepset import SimpleDeepSet


class BaseEncoder(nn.Module):
    """Base encoder specifying minimal init protocol."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


@register_module
class AttentiveLightCurveEmbedding(nn.Module):
    '''Module that performs the embedding for the dependent variables in a time-series'''

    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 128,
                 num_bands: int = 6,
                 band_emb_dim: int = 2,
                 embed_flux_error=False,
                 embed_meta_data=False,
                 embed_time=False,
                 dropout: float = 0.0,
                 scale_embedding_grad_by_freq: bool = False,
                 activation: typing.Union[str, None] = None,
                 band_embedding_files = None,
                 norm= True # Unused
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bands = num_bands
        self.band_emb_dim = band_emb_dim
        self.embed_flux_error = embed_flux_error
        self.embed_meta_data = embed_meta_data
        self.embed_time = embed_time
        self.dropout = nn.Dropout(dropout)
        self.scale_embedding_grad_by_freq = scale_embedding_grad_by_freq
        self.activation = getattr(nn, activation) if activation is not None else activation

        if self.embed_meta_data and not self.embed_flux_error:
            self.embed_meta_data = False
            import sys
            print("WARNING: embed_meta_data is True but embed_flux_error is False, will not perform any embedding", file=sys.stderr)

        self.band_embedding_files = band_embedding_files
        if self.band_embedding_files is not None:
            lambda_grid, filters = map_transmission_filters(self.band_embedding_files, n_points=band_emb_dim)
            band_idx = [BAND_IDX_MAP[key] for key in filters]
            throughput = np.vstack(list(filters.values())).astype(np.float32)
            throughput = torch.tensor(throughput[np.argsort(band_idx)], requires_grad=False)
            self.band_emb = nn.Embedding.from_pretrained(throughput, freeze=True)
            self.num_bands = len(filters)
            self.band_emb_dim = len(lambda_grid)

        else:

            self.band_emb = nn.Embedding(self.num_bands, self.band_emb_dim, scale_grad_by_freq=scale_embedding_grad_by_freq)
            lambda_grid = torch.linspace(0.1, 2.0, self.band_emb_dim, requires_grad=False)[None, None, :, None]
        self.register_buffer("lambda_grid", torch.from_numpy(lambda_grid))


        emb_inputs = self.input_dim
        if self.embed_meta_data:
            self.detection_embedding = nn.Embedding(2, 2)
            emb_inputs += 2
        if self.embed_flux_error:
            self.flux_error_embedding = FPEmbedding(1, self.input_dim)
            emb_inputs += input_dim
        if self.embed_time:
            self.time_embedding = FPEmbedding(1, self.input_dim)
            emb_inputs += input_dim
        self.flux_embedding = FPEmbedding(1, self.input_dim)
        if self.activation is None:
            self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.dropout)
            self.band_to_hidden_dim = nn.Sequential(nn.Linear(2, self.output_dim), self.dropout)
        else:
            self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.activation(), self.dropout)
            self.band_to_hidden_dim = nn.Sequential(nn.Linear(2, self.output_dim), self.activation(), self.dropout)

        self.mht = nn.MultiheadAttention(1, num_heads=1, kdim=self.output_dim, vdim=self.output_dim, batch_first=True, dropout=dropout, add_bias_kv=True)

    @torch.jit.export
    def forward(self, flux, flux_error, band_idx, detection_flag=None, time=None):
        '''
        x : (B, S, K)
            K = 0: time
            K = 1: flux
            K = 2: flux error
            K = 3: Detection Flag
            K = 
        '''

        e_b = self.band_emb(band_idx.squeeze(2))
        emb = self.flux_embedding(flux)

        if self.embed_flux_error:
            e_e = self.flux_error_embedding(flux_error)
            emb = torch.cat([emb, e_e], dim=-1)

        if self.embed_meta_data:
            e_d = self.detection_embedding(detection_flag)
            emb = torch.cat([emb, e_d], dim=-1)

        if self.embed_time:
            e_t = self.time_embedding(time)
            emb = torch.cat([emb, e_t], dim=-1)

        #x = torch.cat([emb, e_b], dim=-1)
        x = self.to_hidden_dim(emb)[:, :, :, None] # B, S, K, 1
        b, s, k, _ = x.shape

        transmission = self.band_to_hidden_dim(torch.cat([self.lambda_grid, e_b[..., None]], dim=-1)) # #B, S, K2, 2
        embedding = self.mht(
            x.reshape(b*s, k, 1), 
            transmission.reshape(b*s, -1, 2), 
            transmission.reshape(b*s, -1, 2),
            need_weights=False)[0].reshape(b, s, k)

        return embedding


@register_module
class ModularLightCurveEmbedding(nn.Module):
    '''and embedding class that you can provide individual modules to.
    This class is more flexible but relies upon the user to correctly configure it
    
    '''
    def __init__(
        self, 
        flux_embedding, 
        aggregator_module,
        flux_error_embedding=None,
        meta_data_embedding=None,
        time_embedding=None,
        band_embedding=None,
        mode='cat',
        output_dim=128,
        ):

        super().__init__()

        assert mode in (
            'cat', 'add'
        )
        self.output_dim = output_dim

        self.flux_embedding = flux_embedding
        self.flux_error_embedding = flux_error_embedding
        self.meta_data_embedding = meta_data_embedding
        self.time_embedding = time_embedding
        self.band_embedding = band_embedding

        self.embed_time = self.time_embedding is not None # Legacy Support
        self.embed_meta_data = self.meta_data_embedding is not None # Legacy Support

        self.aggregator_module = aggregator_module
        
        self.mode = mode

    @torch.jit.export
    def forward(self, flux, flux_error=None, band_idx=None, detection_flag=None, time=None):
        # all inputs assumed to be of shape (B, S, *)

        f_emb = self.flux_embedding(flux)
        f_err_emb = self.flux_error_embedding(flux_error) if self.flux_error_embedding is not None else 0
        md_emb = self.meta_data_embedding(detection_flag) if self.meta_data_embedding is not None else 0
        t_emb = self.time_embedding(time) if self.time_embedding is not None else 0
        b_emb = self.band_embedding(band_idx.squeeze(2)) if self.band_embedding is not None else 0

        if self.mode == 'cat':
            base_emb = torch.cat([emb for emb in [f_emb, f_err_emb, md_emb, t_emb, b_emb] if isinstance(emb, torch.Tensor)], dim=-1)
        else:
            base_emb = f_emb + f_err_emb + md_emb + t_emb + b_emb

        return self.aggregator_module(base_emb)

NUM_BANDS = 6  # keep the same constants you use elsewhere
BAND_EMB_DIM = 2

@register_module
class LightCurveEmbedding(nn.Module):
    '''Module that performs the embedding for the dependent variables in a time-series'''

    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 128,
                 num_bands: int = 6,
                 band_emb_dim: int = 2,
                 embed_flux_error=False,
                 embed_meta_data=False,
                 embed_time=False,
                 dropout: float = 0.0,
                 scale_embedding_grad_by_freq: bool = False,
                 activation: typing.Union[str, None] = None,
                 band_embedding_files = None,
                 norm = False,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bands = num_bands
        self.band_emb_dim = band_emb_dim
        self.embed_flux_error = embed_flux_error
        self.embed_meta_data = embed_meta_data
        self.embed_time = embed_time
        self.dropout = nn.Dropout(dropout)
        self.scale_embedding_grad_by_freq = scale_embedding_grad_by_freq
        self.activation = getattr(nn, activation) if activation is not None else activation
        self.norm = norm

        if self.embed_meta_data and not self.embed_flux_error:
            self.embed_meta_data = False
            import sys
            print("WARNING: embed_meta_data is True but embed_flux_error is False, will not perform any embedding", file=sys.stderr)

        self.band_embedding_files = band_embedding_files
        if self.band_embedding_files is not None:
            lambda_grid, filters = map_transmission_filters(self.band_embedding_files, n_points=band_emb_dim)
            band_idx = [BAND_IDX_MAP[key] for key in filters]
            throughput = np.vstack(list(filters.values())).astype(np.float32)
            throughput = torch.tensor(throughput[np.argsort(band_idx)], requires_grad=False)
            self.band_emb = nn.Embedding.from_pretrained(throughput, freeze=True)
            self.num_bands = len(filters)
            self.band_emb_dim = len(lambda_grid)

        else:

            self.band_emb = nn.Embedding(self.num_bands, self.band_emb_dim, scale_grad_by_freq=scale_embedding_grad_by_freq)
        emb_inputs = self.input_dim + self.band_emb_dim
        if self.embed_meta_data:
            self.detection_embedding = nn.Embedding(2, 2)
            emb_inputs += 2
        if self.embed_flux_error:
            self.flux_error_embedding = FPEmbedding(1, self.input_dim, dropout=0.0)
            emb_inputs += input_dim
        if self.embed_time:
            self.time_embedding = FPEmbedding(1, self.input_dim, dropout=0.0)
            emb_inputs += input_dim
        self.flux_embedding = FPEmbedding(1, self.input_dim, dropout=0.0)
        if norm:
            self.layer_norm = nn.LayerNorm(self.output_dim)
        if self.activation is None:
            if norm:
                self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.layer_norm, self.dropout)
            else:
                self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.dropout)
        else:
            if norm:
                self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.layer_norm, self.activation(), self.dropout)
            else:
                self.to_hidden_dim = nn.Sequential(nn.Linear(emb_inputs, self.output_dim), self.activation(), self.dropout)

        
    @torch.jit.export
    def forward(self, flux, flux_error, band_idx, detection_flag=None, time=None):
        '''
        x : (B, S, K)
            K = 0: time
            K = 1: flux
            K = 2: flux error
            K = 3: Detection Flag
            K = 
        '''

        e_b = self.band_emb(band_idx.squeeze(2))
        emb = self.flux_embedding(flux)

        if self.embed_flux_error:
            e_e = self.flux_error_embedding(flux_error)
            emb = torch.cat([emb, e_e], dim=-1)

        if self.embed_meta_data:
            e_d = self.detection_embedding(detection_flag)
            emb = torch.cat([emb, e_d], dim=-1)

        if self.embed_time:
            e_t = self.time_embedding(time)
            emb = torch.cat([emb, e_t], dim=-1)

        x = torch.cat([emb, e_b], dim=-1)
        x = self.to_hidden_dim(x)

        return x
    
@register_module
class RecurrentODEEncoderV2(nn.Module):
    """
    GRU-based encoder for single-band supernova light curves.
    Encodes [time, normalized flux] sequences into latent vector z.

    V2 has a simplified, but more explicit, interface

    Encoder classes **MUST** have a hidden_dim attribute for proper autoencoder gluing
    """

    def __init__(self, 
                 embedder,
                 rnn,
                 aggregator=None, # Run forward ODE and aggregate with a sequence to embedding network
                 output_network=None, # Run output of aggregator or final hidden state and run a network on it
                 t_max: float=1.0, 
                 N_points: float=50,  
                 autoscale_input_time: bool=True,
                 autoscale_output_time: bool=True,
                 ):
        """
        Args:
            input_dim: number of input features (e.g., time + normalized flux + dummy 1)
            hidden_dim: hidden size of GRU
            latent_dim: dimension of latent z vector
        """
        # NEEDS TO BE BAND AWARE!
        super().__init__()
        self.embedder = embedder
        self.hidden_dim = self.embedder.output_dim
        self.rnn = rnn(hidden_size=self.hidden_dim) if isinstance(rnn, Partial) else rnn
        self.aggregator = aggregator(input_dim=self.hidden_dim+1, hidden_dim=self.hidden_dim) if isinstance(aggregator, Partial) else aggregator
        self.output_network = output_network(ninp=self.hidden_dim, noutp=self.hidden_dim, nhid=self.hidden_dim) if isinstance(output_network, Partial) else output_network
        
        self.t_max = t_max
        self.N_points = N_points
        self.t_grid = None

        self.autoscale_input_time = autoscale_input_time
        self.autoscale_output_time = autoscale_output_time

        if self.autoscale_input_time:
            self.input_time_scaling = nn.Parameter(torch.randn(1)*0+1, requires_grad=True) # Let the network determine the backwards dt
        else:
            self.input_time_scaling = 1.0

        if self.autoscale_output_time:
            self.output_time_scaling = nn.Parameter(torch.randn(1)*0+1, requires_grad=True)
        else:
            self.output_time_scaling = 1.0
        

        self.softplus = nn.Softplus()
        self.compile(mode='max-autotune-no-cudagraphs')


    @torch.jit.export
    def forward(self, x, band_idx, mask, return_sequence=False):
        """
        Args:
            x: (B, T, input_dim) input sequence

        Returns:
            z: sampled latent vector (B, latent_dim)
            z_mean: mean of latent distribution (B, latent_dim)
            z_logvar: log-variance of latent distribution (B, latent_dim)
        """
        if self.t_grid is None: 
            self.t_grid = torch.linspace(0.0, self.t_max, self.N_points, device=x.device)
            if not (self.autoscale_input_time or self.autoscale_output_time):
                self.t_grid = self.t_grid.detach() # Don't backprop through time
        t = x[..., 0]
        t_scale_input = self.softplus(self.input_time_scaling) if self.autoscale_input_time else self.input_time_scaling
        t = t * t_scale_input

        flux = x[..., 1].unsqueeze(-1)
        flux_error = x[..., 2].unsqueeze(-1)
        time = x[..., 0].unsqueeze(-1) if self.embedder.embed_time else None
        detection_flag = x[..., 3].to(torch.int32) if self.embedder.embed_meta_data else None
        x = self.embedder(flux, flux_error, band_idx, detection_flag=detection_flag, time=time) # (B, S, hidden_dim)
        
        h_s, h_n = self.rnn(x, t, mask)  # h_n: (B, hidden_dim), might think of something to do with the hidden sequence since each value is technically a valid observation just shifted in time
        # Could time-shift the t_grids based on the positions in time, then have a (B, S, N_output, hidden_dim) where S_1 is the position of each sequence element and N_output is the forward evolution
        # Should take care since deepsets aggregates over dim=1
        # Time shifting won't have an effect on the forward evolution time grid (Unless all samples evolved to the same point on a common grid)
        # so care would need to be taken in the decoder offsets

        if self.aggregator is not None:
            t_scale_output = self.softplus(self.output_time_scaling) if self.autoscale_output_time else self.output_time_scaling
            t_out = (t_scale_output*self.t_grid).expand(h_n.shape[0], -1) # (B, N_points)
            _, zs = self.rnn.integrator(h_n, t_out) # (B, S, E)
            zs = torch.cat((zs, t_out[:, :, None]), dim=-1) # (B, S, hidden_dim + 1)
            h_n = self.aggregator(zs) # (B, H)
   
        if self.output_network is not None:
            h_n = self.output_network(h_n)
            if return_sequence: # WARNING! This might be huge!
                h_s = self.output_network(h_s)

        if return_sequence:
            return h_n, h_s
        return h_n, None

@register_module
class RecurrentODEEncoder(nn.Module):
    """
    GRU-based encoder for single-band supernova light curves.
    Encodes [time, normalized flux] sequences into latent vector z.
    """

    def __init__(self, 
                 input_dim=3, 
                 hidden_dim=128, 
                 dropout=0.0, 
                 BAND_EMB_DIM=2, 
                 NUM_BANDS=6, 
                 t_max=2.0, 
                 N_points=50, 
                 nlayers=4, 
                 pooling="mean", 
                 norm=True,
                 aggregate_with_deepsets=True,
                 activation_ode='Softplus',
                 activation_deepsets='PReLU',
                 embed_flux_error=False,
                 embed_meta_data=False,
                 bias=True,
                 use_hyper_integrator=False,
                 atol=1.0, rtol=1.0, pcoeff=0.2, icoeff=0.5, 
                 dcoeff=0.0, dt_max=0.5, dt_min=0.0,
                 use_fixed_solver=False,
                 scale_embedding_grad_by_freq: bool = False,
                 autoscale_time: bool=True,
                 use_embedder: bool=False,
                 activation_embedding: typing.Union[str, None]=None,
                 fixed_solver_steps: int = 1):
        """
        Args:
            input_dim: number of input features (e.g., time + normalized flux + dummy 1)
            hidden_dim: hidden size of GRU
            latent_dim: dimension of latent z vector
        """
        # NEEDS TO BE BAND AWARE!
        super().__init__()
        self.hidden_dim = hidden_dim
        self.NUM_BANDS = NUM_BANDS
        self.BAND_EMB_DIM = BAND_EMB_DIM
        self.t_max = t_max
        self.N_points = N_points
        self.nlayers = nlayers
        self.activation_ode = activation_ode
        self.activation_deepsets = activation_deepsets
        self.aggregate_with_deepsets = aggregate_with_deepsets
        self.embed_flux_error = embed_flux_error
        self.embed_meta_data = embed_meta_data
        self.use_hyper_integrator = use_hyper_integrator
        self.use_fixed_solver = use_fixed_solver
        self.autoscale_time = autoscale_time
        self.use_embedder = use_embedder
        self.activation_embedding = activation_embedding
        self.fixed_solver_steps = fixed_solver_steps

        if embed_meta_data and not embed_flux_error:
            import sys
            print("WARNING: embed_meta_data is True but embed_flux_error is False, will not perform any embedding")

        if use_embedder:
            self.embedder = LightCurveEmbedding(input_dim=input_dim,
                                                output_dim=hidden_dim,
                                                num_bands=NUM_BANDS,
                                                band_emb_dim=BAND_EMB_DIM,
                                                embed_flux_error=embed_flux_error,
                                                embed_meta_data=embed_meta_data,
                                                dropout=dropout,
                                                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq,
                                                activation=activation_embedding,)
        else:
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=scale_embedding_grad_by_freq)
            emb_inputs = input_dim + BAND_EMB_DIM
            if self.embed_meta_data:
                self.detection_embedding = nn.Embedding(2, 2)
                emb_inputs += 2
            if self.embed_flux_error:
                self.flux_error_embedding = FPEmbedding(1, input_dim)
                emb_inputs += input_dim
            
            self.flux_embedding = FPEmbedding(1, input_dim)
            self.linear_layer = nn.Linear(emb_inputs, hidden_dim)

        self.rnn = RNNODEEncoder(
            hidden_dim,
            batch_first=True,
            bias=bias,
            nlayers=nlayers,
            norm=norm,
            activation=activation_ode,
            dropout=dropout,
            use_hyper_integrator=use_hyper_integrator,
            atol=atol, rtol=rtol, pcoeff=pcoeff, icoeff=icoeff, 
            dcoeff=dcoeff, dt_max=dt_max, dt_min=dt_min,
            use_fixed_solver=use_fixed_solver,
            fixed_solver_steps=self.fixed_solver_steps
        )
        #self.rnn = torch.compile(self.rnn)

        

        #additional tests
        
        #self.to_latent_timeseries = torch.compile(self.to_latent_timeseries)
        #self.deepset = SimpleDeepSet(hidden_dim, hidden_dim, activation='PReLU', dropout=dropout)
        # Want to include times
        if self.aggregate_with_deepsets:
            self.to_latent_timeseries = ODEDecoder(hidden_size=hidden_dim, activation=activation_ode, batch_first=True)
            self.to_latent_timeseries.integrator = self.rnn.integrator
            self.deepset = SimpleDeepSet(hidden_dim+1, 
                                        hidden_dim, 
                                        activation=activation_deepsets, 
                                        dropout=dropout, 
                                        pooling=pooling,
                                        norm=norm)
        self.t_grid = None
        if self.autoscale_time:
            self.time_scaling = nn.Parameter(torch.randn(1), requires_grad=True) # Let the network determine the dt
        else:
            self.time_scaling = 1.0
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

        if not self.use_hyper_integrator:
            self.compile(mode='max-autotune-no-cudagraphs')


    @torch.jit.export
    def forward(self, x, band_idx, mask, time_sorted=None, sequence_batch_mask=None):
        """
        Args:
            x: (B, T, input_dim) input sequence

        Returns:
            z: sampled latent vector (B, latent_dim)
            z_mean: mean of latent distribution (B, latent_dim)
            z_logvar: log-variance of latent distribution (B, latent_dim)
        """
        if self.t_grid is None:
            self.t_grid = torch.linspace(0.0, self.t_max, self.N_points, device=x.device)#.detach() # (S)

        t = x[..., 0]
        t_scale = self.softplus(self.time_scaling) if self.autoscale_time else self.time_scaling
        t = t * t_scale

        if self.use_embedder:
            flux = x[..., 1].unsqueeze(-1)
            flux_error = x[..., 2].unsqueeze(-1)
            detection_flag = x[..., 3].to(torch.int32) if self.embed_meta_data else None
            x = self.embedder(flux, flux_error, band_idx, detection_flag=detection_flag)
        else:
            e_b = self.band_emb(band_idx.squeeze(2))
            f = x[..., 1:2]
            emb = self.flux_embedding(f) # 

            if self.embed_flux_error:
                f = x[..., 2:3]
                e_e = self.flux_error_embedding(f)
                emb = torch.cat([emb, e_e], dim=-1)

            if self.embed_meta_data:
                d = x[..., 3].to(torch.int32)
                e_d = self.detection_embedding(d)
                emb = torch.cat([emb, e_d], dim=-1)

            x = torch.cat([emb, e_b], dim=-1)
            x = self.linear_layer(x)
            x = self.dropout(x)
        
        _, h_n = self.rnn(x, t, mask, time_sorted, sequence_batch_mask)  # h_n: (B, hidden_dim)

        if self.aggregate_with_deepsets:
            if self.N_points == 1:
                zs = h_n[:, None, :]
            else:
                _, zs = self.to_latent_timeseries(t_scale*self.t_grid, h_n) # (B, S, E)
            # concat the time points!
            zs = torch.cat((zs, t_scale*self.t_grid.expand(zs.shape[0], -1)[:, :, None]), dim=-1)
            z = self.deepset(zs) # (B, H)
            return z
        return h_n


@register_module
class GRUEncoder(nn.Module):
    """
    GRU-based encoder for single-band supernova light curves.
    Encodes [time, normalized flux] sequences into latent vector z.
    """

    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1, dropout=0.0):
        """
        Args:
            input_dim: number of input features (e.g., time + normalized flux + dummy 1)
            hidden_dim: hidden size of GRU
            latent_dim: dimension of latent z vector
        """
        # NEEDS TO BE BAND AWARE!
        super().__init__()
        self.hidden_dim = hidden_dim

        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=True)
        self.time_embedding = LinearTimeEmbedding(emb_dim=input_dim)
        self.float_embedding = FPEmbedding(1, input_dim)
        self.gru = nn.GRU(
            input_dim + BAND_EMB_DIM,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            bias=False,
            dropout=dropout,
        )

        #self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        #self.fc_logvar = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x, band_idx, mask):
        """
        Args:
            x: (B, T, input_dim) input sequence

        Returns:
            z: sampled latent vector (B, latent_dim)
            z_mean: mean of latent distribution (B, latent_dim)
            z_logvar: log-variance of latent distribution (B, latent_dim)
        """
        e_b = self.band_emb(band_idx[:, :, 0])
        t = x[..., :1]
        t_emb = self.time_embedding(t)
        f = x[..., 1:2]
        f_emb = self.float_embedding(f)
        x = f_emb + t_emb
        x = torch.cat([x, e_b], dim=-1)
        x[~mask] = 0.0
        _, h_n = self.gru(x)  # h_n: (1, B, hidden_dim)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)

        
        return h_n


@register_module
class GRUMaskedEncoder(nn.Module):
    """
    GRU-based encoder for multiband supernova light curves.
    Encodes [time, normalized flux, scalar_flux] + band embedding
    into a memory vector using masking to ignore padded time steps.
    """

    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM)
        self.gru = nn.GRU(
            input_size=input_dim + BAND_EMB_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            bias=False,
        )

    def forward(self, x: torch.Tensor, band_idx: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x         : (B, T, 3) - input features per time step [time, flux, scalar_norm]
            band_idx  : (B, T, 1) - int band index per time step
            pad_mask  : (B, T)    - bool mask (True = valid, False = padded)

        Returns:
            memory    : (B, hidden_dim) - per-curve summary memory vector
        """
        B, T, _ = x.shape

        # ── 1. Embed bands and concatenate with x
        e_b = self.band_emb(band_idx[:, :, 0])    # (B, T, BAND_EMB_DIM)
        x_cat = torch.cat([x, e_b], dim=-1)       # (B, T, input_dim + BAND_EMB_DIM)

        # ── 2. Compute actual lengths from pad_mask (sum of valid tokens)
        lengths = pad_mask.sum(dim=1).to(torch.int64)  # (B,)

        # ── 3. Sort by length for packing (required by some PyTorch versions)
        # pack_padded_sequence requires that sequences be sorted in decreasing order of length if enforce_sorted=True.
        lengths, perm_idx = lengths.sort(descending=True)
        x_cat = x_cat[perm_idx]
        
        # ── 4. Pack the sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_cat, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # ── 5. GRU encoding
        _, h_n = self.gru(x_packed)   # h_n: (num_layers, B, hidden_dim)
        h_n = h_n[-1]                 # Final layer's hidden state → (B, hidden_dim)

        # ── 6. Undo the permutation to restore original order
        _, unperm_idx = perm_idx.sort()
        h_n = h_n[unperm_idx]         # (B, hidden_dim)

        return h_n
