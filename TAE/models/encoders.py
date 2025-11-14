import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from TAE.util import register_module
from TAE.models.embeddings import LinearTimeEmbedding, FPEmbedding, PlotterEmbedding, PositionalEncoding
from TAE.models.ModelNN import ResNet
from TAE.models.neuralode import RNNODEEncoder, ODEDecoder
from TAE.models.deepset import SimpleDeepSet


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

NUM_BANDS = 6  # keep the same constants you use elsewhere
BAND_EMB_DIM = 2


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
                 scale_embedding_grad_by_freq: bool = False):
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

        if embed_meta_data and not embed_flux_error:
            import sys
            print("WARNING: embed_meta_data is True but embed_flux_error is False, will not perform any embedding")

        self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=scale_embedding_grad_by_freq)
        emb_inputs = input_dim + BAND_EMB_DIM
        if self.embed_meta_data:
            self.detection_embedding = nn.Embedding(2, 2)
            emb_inputs += 2
        if self.embed_flux_error:
            self.flux_error_embedding = FPEmbedding(1, input_dim)
            emb_inputs += input_dim
        
        self.flux_embedding = FPEmbedding(1, input_dim)
        self.rnn = RNNODEEncoder(
            hidden_dim,
            batch_first=True,
            bias=bias,
            nlayers=nlayers,
            norm=norm,
            activation=activation_ode,
            use_hyper_integrator=use_hyper_integrator,
            atol=atol, rtol=rtol, pcoeff=pcoeff, icoeff=icoeff, 
            dcoeff=dcoeff, dt_max=dt_max, dt_min=dt_min,
            use_fixed_solver=use_fixed_solver
        )
        #self.rnn = torch.compile(self.rnn)

        self.linear_layer = nn.Linear(emb_inputs, hidden_dim)

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
        #self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        #self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.time_scaling = nn.Parameter(torch.randn(1), requires_grad=True) # Let the network determine the dt
        self.softplus = nn.Softplus()

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

        '''
        t = x[..., 0]
        f = x[..., 1]
        b = band_idx.squeeze(2)

        x = self.embedder(t, f, b, mask)
        '''
        #################
        e_b = self.band_emb(band_idx.squeeze(2))
        t = x[..., 0]
        t_scale = self.softplus(self.time_scaling)
        t = t * t_scale

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
        #x[~mask] = 0.0
        #x_old = x
        #x = x.masked_fill_(~mask.unsqueeze(-1), 0.0) #Should validate that unfilled gives back the same embedding
        #################
        

        _, h_n = self.rnn(x, t, mask, time_sorted, sequence_batch_mask)  # h_n: (B, hidden_dim)
        #_, h_n2 = self.rnn(x_old, t, mask, time_sorted, sequence_batch_mask)
        #assert (h_n == h_n2).all(), f"Outputs Don't Match!: {h_n},{h_n2}"
        #h_n = h_n.squeeze(0)  # (B, hidden_dim)
        #return h_n

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
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        NUM_BANDS: int = 6,
        activation: str = 'PReLU'
    ):
        super().__init__()

        self.num_bands = NUM_BANDS
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.band_emb = nn.Embedding(NUM_BANDS, input_dim, scale_grad_by_freq=True)
        self.time_embedding = LinearTimeEmbedding(emb_dim=input_dim)
        self.float_embedding = FPEmbedding(1, input_dim)
        self.activation = getattr(nn, activation)

        self.transformer = nn.Transformer(
            d_model=input_dim, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.position_embeddings = nn.Parameter(torch.randn(size=(1, 1, input_dim)), requires_grad=True)

        self.embedder = PlotterEmbedding(N_width=100, N_height=input_dim, gamma=1.0, output_dim=input_dim) # NEW!

        self.encoder = self.transformer.encoder
        #self.phi_out = nn.Sequential(
        #    nn.Linear(input_dim, hidden_dim),
        #    self.activation(),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    self.activation(),
        #)

        self.phi_out = nn.Sequential(
            ResNet(input_dim, hidden_dim, nlayers=2, nhid=hidden_dim, activation=activation, dropout=dropout),
            self.activation(),
        )

        self.attn_mask = None



    def forward(self, x, band_idx, mask, time_sorted=None, sequence_batch_mask=None):
        """
        Args:
            x: (B, T, input_dim) input sequence

        Returns:
            z: sampled latent vector (B, latent_dim)
            z_mean: mean of latent distribution (B, latent_dim)
            z_logvar: log-variance of latent distribution (B, latent_dim)
        """
        '''
        e_b = self.band_emb(band_idx[:, :, 0])
        t = x[..., :1]
        t_emb = self.time_embedding(t)
        f = x[..., 1:2]
        f_emb = self.float_embedding(f)
        x = f_emb + t_emb + e_b
        x[~mask] = 0.0
        '''
        t = x[..., 0]
        f = x[..., 1]
        b = band_idx.squeeze(2)
        x = self.embedder(t, f, b, mask) + self.position_embeddings
        if self.attn_mask is None:
            self.attn_mask = self.transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        memory = self.encoder(x, mask=self.attn_mask, is_causal=True)  # h_n: (B, S, input_dim)
        memory = (self.phi_out(memory)*mask.unsqueeze(-1)).sum(1)

        return memory

        return z, z_mean, z_logvar


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
