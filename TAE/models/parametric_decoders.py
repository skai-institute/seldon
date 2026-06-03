import numpy as np
import torch
from torch import nn
from TAE.util import register_module, map_transmission_filters
import torch
from TAE.models.ModelNN import ResNet, ResidualBlock, ResidualNetwork, Partial
from torch.nn import functional as F
from TAE.models.ModelNN import ExponentialDistriubtionQuantileMapping, NormalDistriubtionQuantileMapping, SlidingBasisSuppression
import typing
from TAE.models.basis import DoubleSigmoid, SpectrumBasis, FullSpectrumBasis
from TAE.models.embeddings import PositionalEncoding
import torch.nn.init as init
from TAE.datasets.utils import BAND_IDX_MAP

@register_module
class SmallParametricDecoder(nn.Module):
    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        N=6,
        DF=3,
        activation: str = "GELU",
        dropout: float = 0.0,
        nlayers=4,
        norm: bool = False,
    ):
        super().__init__()
        self.K = K # Number of basis functions
        self.N = N # Number of sets of basis functions
        self.DF = DF  # the number of parameters each sigmoidal basis function has
        #self.hidden = hidden
        self.hidden = hidden
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.nlayers = nlayers
        self.hidden
        self.expansion = [nn.Linear(latent, self.hidden)]
        if norm:
            self.expansion.append(nn.LayerNorm(self.hidden))
        self.expansion.extend([self.activation(), nn.Dropout(p=dropout)])
        self.expansion = nn.Sequential(*self.expansion)
        self.mlp = []
        # The following needs some transposition
        # in_channels -> n_networks * n_hidden
        # n_hidden ~> K * DF
        # groups -> n_networks
        # Could try to expand this to have the convolution run over N, K for variable length encodings
        
        for _ in range(nlayers-1):
            mlp = [nn.Linear(self.hidden, self.hidden)]
            if norm:
                mlp.append(nn.LayerNorm(self.hidden))
            mlp.extend([self.activation(), nn.Dropout(p=dropout)])
            self.mlp.append(ResidualBlock(*mlp))
        output_layer = nn.Linear(self.hidden, self.DF)
        self.mlp.append(output_layer)
        self.mlp = nn.Sequential(*self.mlp)

        self.pos_encoding = PositionalEncoding(self.hidden, dropout=dropout)
        
        # DF are independent, N are dependent, K are independent

    def forward(self, z, *args):  # z:(B,E)

        params = self.expansion(z)[:, None, :] # (B, 1, H) 
        params = self.pos_encoding(params.expand(-1, self.N*self.K, -1)).reshape(-1, self.N, self.K, self.hidden) # essentially a patch embedding
        params = self.mlp(params).reshape(-1, 1, self.N, self.K, self.DF) # (B, 1, N, K, DF)
        return params  # (B, 1, N, K, DF)

@register_module
class ParametricDecoder(nn.Module):
    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        N=6,
        DF=3,
        activation: str = "GELU",
        dropout: float = 0.0,
        nlayers=4,
        norm: bool = False,
    ):
        super().__init__()
        self.K = K # Number of basis functions
        self.N = N # Number of sets of basis functions
        self.DF = DF  # the number of parameters each sigmoidal basis function has
        #self.hidden = hidden
        self.hidden = hidden
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.nlayers=nlayers
        M = self.N*self.hidden
        self.expansion = [nn.Linear(latent, M)]
        if norm:
            self.expansion.append(nn.LayerNorm(M))
        self.expansion.extend([self.activation(), nn.Dropout(p=dropout)])
        self.expansion = nn.Sequential(*self.expansion)
        self.mlp = []
        # The following needs some transposition
        # in_channels -> n_networks * n_hidden
        # n_hidden ~> K * DF
        # groups -> n_networks
        # Could try to expand this to have the convolution run over N, K for variable length encodings
        
        for _ in range(nlayers-1):
            mlp = [nn.Conv1d(M, M, groups=self.N, kernel_size=1)]
            if norm:
                mlp.append(nn.GroupNorm(self.N, M))
            mlp.extend([self.activation(), nn.Dropout(p=dropout)])
            self.mlp.append(ResidualBlock(*mlp))
        output_layer = nn.Conv1d(M, self.N*self.DF*self.K, groups=self.N, kernel_size=1)
        self.mlp.append(output_layer)
        self.mlp = nn.Sequential(*self.mlp)
        
        # DF are independent, N are dependent, K are independent

    def forward(self, z, *args):  # z:(B,E)


        params = self.expansion(z).unsqueeze(-1) # (B, HxKxDF, 1) 
        params = self.mlp(params).reshape(-1, 1, self.N, self.K, self.DF) # (B, 1, N, K, DF)
        return params  # (B, 1, N, K, DF)


@register_module
class DecoupledParametricDecoder(nn.Module):

    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        NUM_BANDS=6,
        BAND_EMB_DIM=4,
        DF=3,
        nlayers=4,
        activation: str = "GELU",
        dropout: float = 0.0,
        decouple_dim: typing.Optional[typing.Union[typing.List, int, None]] = None,
        band_dot_prod: bool = False,
        norm: bool = False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        scale_embedding_grad_by_freq: bool = False,
    ):
        
        super().__init__()
        
        if isinstance(decouple_dim, list):
            if len(decouple_dim) == 0:
                decouple_dim = []
        elif decouple_dim is None:
            decouple_dim = []
        else:
            raise TypeError(f'decouple_dim must be list or None but got {type(decouple_dim)}:{decouple_dim}')
        
        decouple_dim.append(latent - sum(decouple_dim))
        
        self.decouple_dim = decouple_dim
        self.num_decoders = len(decouple_dim)
        self.degrees_of_freedom = [1]*(self.num_decoders - 1) + [DF - (self.num_decoders - 1)]

        self.nlayers = nlayers
        self.rescale_decoupled_network = rescale_decoupled_network

        # Do some checks on the parameters
        if isinstance(self.decouple_dim, list):
            assert sum(self.decouple_dim) == latent, f"total decouple dim ({self.decouple_dim}) is != latent dim ({latent})!"
            assert len(self.decouple_dim) < DF, f"number of decoupling parameters ({len(self.decouple_dim)}) is >= degrees of freedom ({DF})!"

        
        self.decoders = nn.ModuleList()
        N = self.num_decoders
        hidden_dim = hidden
        for dof, input_dim in zip(self.degrees_of_freedom, self.decouple_dim):
            if self.rescale_decoupled_network:
                hidden_dim = int(hidden * np.sqrt((input_dim) / latent)) # Consider sqrt vs linear
            #self.decoders.append(
            #    ParametricDecoder(
            #        latent=input_dim,
            #        hidden=hidden_dim,
            #        K=K,
            #        N=NUM_BANDS,
            #        DF=dof,
            #        activation=activation,
            #        dropout=dropout,
            #        nlayers=self.nlayers,
            #        norm=norm,
            #        )
            #)
            self.decoders.append(
                BandParametricDecoder(
                latent=input_dim,
                hidden=hidden_dim,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=dof,
                activation=activation,
                dropout=dropout,
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
                )
            )
        self.compile(mode='max-autotune')
            
    def forward(self, z, t, band_idx):

        outputs = []
        start = 0
        for decoder, input_dim in zip(self.decoders, self.decouple_dim):
            outputs.append(decoder(z[..., start:start+input_dim], t, band_idx))
            start += input_dim

        outputs = torch.cat(outputs, dim=-1)

        return outputs


@register_module
class GeneralizedDecoupledBandParametricDecoder(nn.Module):

    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        NUM_BANDS=6,
        BAND_EMB_DIM=4,
        DF=3,
        nlayers=4,
        activation: str = "GELU",
        dropout: float = 0.0,
        decouple_dim: typing.Optional[typing.Union[typing.List, int, None]] = None,
        band_dot_prod: bool = False,
        norm: bool = False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        scale_embedding_grad_by_freq: bool = False,
    ):
        super().__init__()
        if isinstance(decouple_dim, list):
            if len(decouple_dim) == 0:
                decouple_dim = None
            elif len(decouple_dim) == 1:
                decouple_dim = decouple_dim[0]
        self.decouple_dim = decouple_dim
        self.band_dot_prod = band_dot_prod
        self.inp_dim = 0
        self.nlayers = nlayers
        self.rescale_decoupled_network = rescale_decoupled_network
        self.sed_mode = sed_mode

        # Do some checks on the parameters
        if isinstance(self.decouple_dim, list):
            assert sum(self.decouple_dim) < latent, f"total decouple dim ({self.decouple_dim}) is >= latent dim ({latent})!"
            assert len(self.decouple_dim) < DF, f"number of decoupling parameters ({len(self.decouple_dim)}) is >= degrees of freedom ({DF})!"
            

        if self.decouple_dim is None or self.decouple_dim == 0:
            self.first_decoder = None
            self.inp_dim = 0
            self.decoder = BandParametricDecoder(
                latent=latent,
                hidden=hidden,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
                )
        elif isinstance(self.decouple_dim, int):
            self.first_decoder = None
            self.inp_dim = self.decouple_dim
            self.decoder = DecoupledBandParametricDecoder(
                latent=latent,
                hidden=hidden,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                decouple_dim=self.decouple_dim,
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq)
        
        else:
            self.inp_dim = self.decouple_dim[0]+self.decouple_dim[1]# gives back 2 decoupled slices on the first dim
            if rescale_decoupled_network:
                d_hidden_dim = int(hidden * np.sqrt(self.inp_dim / latent)) # Make this go as sqrt?  
                c_hidden_dim = int(hidden * np.sqrt((latent - self.inp_dim) / latent))
                assert d_hidden_dim > 0, f"Rescaled decoupled dimension is too small! latent:{latent}, decouple_dim:{self.inp_dim}, hidden_dim:{hidden}"
            else:
                d_hidden_dim = hidden
                c_hidden_dim = hidden
            self.first_decoder = DecoupledBandParametricDecoder( 
                latent=self.inp_dim,
                hidden=d_hidden_dim,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=2,
                activation=activation,
                dropout=dropout,
                decouple_dim=self.decouple_dim[0],
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq)
            self.decoder = GeneralizedDecoupledBandParametricDecoder(
                latent=latent - self.inp_dim,
                hidden=c_hidden_dim,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF-2,
                activation=activation,
                dropout=dropout,
                decouple_dim=self.decouple_dim[2:],
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )

    def forward(self, z, t, band_idx):

        out = self.decoder(z[..., self.inp_dim:], t, band_idx)
        if self.first_decoder is not None:
            p_1 = self.first_decoder(z[..., :self.inp_dim], t, band_idx)
            out = torch.cat((p_1, out), dim=-1)

        return out


@register_module
class DecoupledBandParametricDecoder(nn.Module): # TODO: Generalize this (amp invariance may be important)

    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        NUM_BANDS=6,
        BAND_EMB_DIM=4,
        DF=3,
        activation: str = "GELU",
        dropout: float = 0.0,
        decouple_dim: int = 1,
        nlayers=4,
        band_dot_prod: bool = False,
        norm: bool = False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        scale_embedding_grad_by_freq: bool = False,
    ):
        super().__init__()
        self.decouple_dim = decouple_dim
        self.band_dot_prod = band_dot_prod
        self.nlayers = nlayers
        self.rescale_decoupled_network = rescale_decoupled_network
        self.sed_mode = sed_mode

        if rescale_decoupled_network:
            d_hidden_dim = int(hidden * np.sqrt(self.decouple_dim / latent)) # Make this go as sqrt or linear space?  
            c_hidden_dim = int(hidden * np.sqrt((latent - self.decouple_dim) / latent))
            assert d_hidden_dim > 0, f"Rescaled decoupled dimension is too small! latent:{latent}, decouple_dim:{self.decouple_dim}, hidden_dim:{hidden}"
        else:
            d_hidden_dim = hidden
            c_hidden_dim = hidden

        self.to_time = BandParametricDecoder(
            latent=self.decouple_dim,
            hidden=d_hidden_dim,
            K=K,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=1,
            activation=activation,
            dropout=dropout,
            band_dot_prod=band_dot_prod,
            nlayers=self.nlayers,
            norm=norm,
            sed_mode=sed_mode,
            scale_embedding_grad_by_freq=scale_embedding_grad_by_freq)
        self.to_params = BandParametricDecoder(
            latent=latent-self.decouple_dim,
            hidden=c_hidden_dim,
            K=K,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=DF-1,
            activation=activation,
            dropout=dropout,
            band_dot_prod=band_dot_prod,
            nlayers=self.nlayers,
            norm=norm,
            sed_mode=sed_mode,
            scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
        )

    def forward(self, z, t, band_idx):

        p_t = self.to_time(z[..., :self.decouple_dim], t, band_idx)
        p_other = self.to_params(z[..., self.decouple_dim:], t, band_idx)
        out = torch.cat((p_t, p_other), dim=-1)
        return out



class BandParametricDecoder(nn.Module):
    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        NUM_BANDS=6,
        BAND_EMB_DIM=4,
        DF=3,
        activation: str = "GELU",
        dropout: float = 0.0,
        nlayers=4,
        band_dot_prod = False,
        norm: bool = False,
        sed_mode: bool = False,
        scale_embedding_grad_by_freq: bool = False,
    ):
        super().__init__()
        self.K = K
        self.NUM_BANDS = NUM_BANDS
        self.BAND_EMB_DIM = BAND_EMB_DIM
        self.DF = DF  # the number of parameters each sigmoidal basis function has
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.band_dot_prod = band_dot_prod
        self.nlayers=nlayers
        self.sed_mode = sed_mode
        if self.sed_mode and self.band_dot_prod:
            import sys
            print("WARNING: both `sed_mode` and `band_dot_prod` are true.  Will only use SED mode", file=sys.stderr)
        if self.BAND_EMB_DIM > 0:
            self.band_emb = nn.Embedding(self.NUM_BANDS, self.BAND_EMB_DIM, scale_grad_by_freq=scale_embedding_grad_by_freq)

        if self.band_dot_prod or self.sed_mode:
            assert self.BAND_EMB_DIM > 0, "Cannot perform a dot product without a band embedding"
            inp = latent # multiply and sum z @ e_b
            self.mlp = ResNet(inp, BAND_EMB_DIM * self.DF * K, nlayers=self.nlayers, nhid=hidden, activation=activation, dropout=dropout, norm=norm)
        else:
            inp = latent + BAND_EMB_DIM  # concat z‖e_b
            self.mlp = ResNet(inp, NUM_BANDS * self.DF * K, nlayers=self.nlayers, nhid=hidden, activation=activation, dropout=dropout, norm=norm)
 
        self.compile()

    def forward(self, z, t, band_idx):  # t:(B,T,1)

        #z = z.unsqueeze(1).expand(z.shape[0], band_idx.shape[1], z.shape[1]) # (B, S, E)
        if self.BAND_EMB_DIM > 0:
            z = z#.unsqueeze(1).expand(z.shape[0], band_idx.shape[1], z.shape[1]) # (B, S, E), maybe do this after mlp
            e_b = self.band_emb(band_idx[:, :, 0])  # (B,seq,E)
            if not self.band_dot_prod and not self.sed_mode:
                z = torch.cat([z, e_b], dim=-1)
            params = self.mlp(z) # (B, 3K) # Should be (B, band, 3k), apply same mlp for each E in the sequnce (clones)
            params = params.unsqueeze(1).expand(params.shape[0], band_idx.shape[1], params.shape[1])
        else:
            if not self.band_dot_prod:
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], band_idx.shape[1], -1)
            else:
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], self.BAND_EMB_DIM, -1)
        
        # import pdb; pdb.set_trace()
        #params = self.mlp(z) # (B, 3K) # Should be (B, band, 3k)
        
        if not self.band_dot_prod and not self.sed_mode:
            params = params.view(
                params.shape[0], params.shape[1], self.NUM_BANDS, self.DF * self.K
            )  # (B,band,3K)
            params = (
                torch.gather(
                    params, 2, band_idx.unsqueeze(2).expand(-1, -1, 1, self.DF * self.K)
                )
                .squeeze(2)
                .view(params.shape[0], -1, self.K, self.DF)
            )  # (B, K, 3)
        else:
            if not self.sed_mode:
                e_b = e_b.unsqueeze(-1).expand(-1, -1, -1, self.DF * self.K) # (B, S, E, DF x K)
                params = params.view(
                    params.shape[0], params.shape[1], self.BAND_EMB_DIM, self.DF * self.K
                )  # (B, S ,E , DF x K)
                params = (params * e_b).sum(2) # (B, S, DF x K)
                params = params.view(params.shape[0], -1, self.K, self.DF) # (B, S, K, 3)
            else:
                params = params.view(
                    params.shape[0], params.shape[1], self.BAND_EMB_DIM, self.K, self.DF
                ) # (B, S, E, K, DF)


        return params  # (B, S, K, DF)

@register_module
class BazinDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden=128, NUM_BANDS=6, BAND_EMB_DIM=4, activation="GELU", dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.to_params = BandParametricDecoder(
            latent_dim, hidden, 1, NUM_BANDS, BAND_EMB_DIM, 4, activation=activation,
            dropout=dropout
        )
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx):
        # A_raw, t0_raw, log_tr, log_tf = self.to_params(z, t, band_idx).unbind(-1)
        params = self.to_params(z, t, band_idx)
        A_raw = params[:, :, 0, 1]
        t0_raw = params[:, :, 0, 0]
        log_tr = params[:, :, 0, 2]
        log_tf = params[:, :, 0, 3]

        A = self.softplus(A_raw) + 1e-3
        t0 = t0_raw.sigmoid()
        τ_r = 0.02 + self.softplus(log_tr)
        τ_f = 0.05 + self.softplus(log_tf)

        t_shift = t - t0.unsqueeze(-1)
        num = torch.exp(-t_shift / τ_f.unsqueeze(-1))
        denom = 1 + torch.exp(-t_shift / τ_r.unsqueeze(-1))
        f = A.unsqueeze(-1) * num / denom
        return f

# ---------------------------------------------------------------------------
#  Sigmoid-basis decoder  (with mild dropout regularisation)
# ---------------------------------------------------------------------------
@register_module
class FunctionalBasisDecoder(nn.Module):

    def __init__(self, latent_dim=128, hidden=128, num_basis=8, NUM_BANDS=6, activation="GELU", dropout=0.0, basis_function=""):

        super().__init__()
        self.num_basis = num_basis
        self.to_params = BandParametricDecoder(
            latent=latent_dim,
            hidden=hidden,
            K=num_basis,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=4,
            DF=3,
            activation=activation,
            dropout=dropout,              # first dropout inside MLP
        )
        self.basis_func = getattr(F, basis_function)

    def forward(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        t: torch.Tensor,  # (B, T, 1)
        band_idx: torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:  # → (B, T, 1)

        # (B, T, K, 3)  → last dim = (w_raw, μ_raw, log σ)
        params = self.to_params(z, t, band_idx)
        t0, A, k = params.unbind(-1)  # each (B, T, K)

        # Broadcast time to K bases
        t_expanded = t.expand(-1, -1, self.num_basis)  # (B, T, K)

        basis = self.basis_func((t_expanded - t0) * k)  # (B, T, K)

        # Weighted sum of bases → (B, T, 1)
        out = torch.sum(A * basis, dim=-1, keepdim=True)
        return out

@register_module
class GaussianBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        decoupled=False, decouple_dim=1,
        band_dot_prod=False,
        nlayers=4,
        logscale_output=True,
        norm=False,
        suppress_higher_terms=False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        accumulate: bool = False,
        scale_embedding_grad_by_freq: bool = False
    ):
        super().__init__()
        self.band_dot_prod = band_dot_prod
        self.num_basis = num_basis
        self.multiple_decouple = isinstance(decouple_dim, list)
        self.nlayers=nlayers
        self.logscale_output = logscale_output
        self.suppress_higher_terms = suppress_higher_terms
        self.sed_mode=sed_mode
        self.band_emb_dim = BAND_EMB_DIM
        self.accumulate = accumulate
        if accumulate:
            import sys
            print("WARNING: Parameters Accumulation not Implimeted for Gaussian Decoder", file=sys.stderr)
        DF = 4
        if self.multiple_decouple:
            DF += 1
        # DF = 3 → (w_raw, μ_raw, log σ)
        if decoupled:
            self.to_params = GeneralizedDecoupledBandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                decouple_dim=decouple_dim,
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        else:
            self.to_params = BandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                band_dot_prod=band_dot_prod,
                nlayers=self.nlayers,
                norm=norm,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        self.softplus = nn.Softplus()
        self.exp_map = ExponentialDistriubtionQuantileMapping(init=0.25)
        self.normal_map = NormalDistriubtionQuantileMapping(sigma_init=0.1, mu_init=0.0)

        if self.multiple_decouple:
            self.exp_map_offset = ExponentialDistriubtionQuantileMapping(init=8.0)
            self.normal_map_offset = NormalDistriubtionQuantileMapping(sigma_init=0.1, mu_init=2.0)

        if self.suppress_higher_terms:
            self.supression_parameters = nn.Parameter(torch.randn(size=(1, 1, num_basis)), requires_grad=True)
            self.suppressor_activation = nn.Hardtanh(min_val=0.0, max_val=0.0, min_value=0.0, max_value=1.0)

        if self.sed_mode:
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=False)

    def forward(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        t: torch.Tensor,  # (B, T, 1)
        band_idx: torch.Tensor,  # (B, T, 1)
        return_parameters: bool=False,
    ) -> torch.Tensor:  # → (B, T, 1)

        # (B, T, K, 3)  → last dim = (w_raw, μ_raw, log σ)
        params = self.to_params(z, t, band_idx)
        if self.multiple_decouple:
            mu_offset, w_offset, w_raw, mu_raw, sigma = params.unbind(-1)

            w_offset = self.exp_map_offset(w_offset)
            mu_offset = self.normal_map_offset(mu_offset)

            w_offset = torch.mean(w_offset, dim=-1, keepdim=True)
            w_raw = self.exp_map(w_raw) # weights are on an exponential distribution
            weights = w_raw * w_offset  / torch.mean(w_raw, dim=-1, keepdim=True).clamp(min=1e-6) # - torch.mean(w_raw, dim=-1, keepdim=True)
        else:
            mu_offset, mu_raw, w_raw, sigma = params.unbind(-1)  # each (B, T, K)
            weights = self.exp_map(w_raw) # weights are on an exponential distribution
        mu_offset = torch.mean(mu_offset, dim=-1, keepdim=True)
        mu_raw = self.normal_map(mu_raw)
        mu_raw = mu_raw + mu_offset - torch.mean(mu_raw, dim=-1, keepdim=True)

        # Centre positions lie in [0, t_max] per light curve
        mu = mu_raw

        # Broadcast time to K bases
        if self.sed_mode:
            t_expanded = t[:, :, None, :].expand(-1, -1, self.band_emb_dim, self.num_basis) # (B, T, E, K)
            e_b = self.band_emb(band_idx[:, :, 0])[..., None]**2
        else:
            t_expanded = t.expand(-1, -1, self.num_basis) # (B, T, K)

        basis = torch.exp(-(((t_expanded - mu) * sigma).square()).clamp(min=-20.0, max=20.0))
        #weights = w_raw - torch.mean(w_raw, dim=-1, keepdim=True)  # (B, T, K)
        
        #weights = w_raw.square()
        if self.suppress_higher_terms: # This allows the network to learn exactly how many basis functions to include
            weights = weights * self.suppressor_activation(self.softplus(self.supression_parameters).cumsum(-1))


        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()

        # Weighted sum of bases → (B, T, 1)
        f = torch.mean(weights * basis, dim=-1, keepdim=True)
        if self.sed_mode: # Do a (normalized) dot-product over the SED at each band location
            f = (e_b * f).sum(dim=-2) / (e_b).sum(dim=-2)

        if self.logscale_output:
            f = torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5
        else:
            f = f - 0.5

        if return_parameters:
            return (f, None)

        return f

@register_module
class SigmoidBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        decoupled=False, decouple_dim=1,
        nlayers=4,
        band_dot_prod=False,
        logscale_output=True,
        suppress_higher_terms=False,
        norm=False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        accumulate: bool = False,
        include_redshift: bool = False,
        tanhshrink_centroids: bool = False,
        autoscale_outputs: bool = False,
        scale_embedding_grad_by_freq: bool = False
    ):
        super().__init__()
        self.num_basis = num_basis
        self.band_dot_prod = band_dot_prod
        self.nlayers = nlayers
        self.logscale_output = logscale_output
        self.suppress_higher_terms = suppress_higher_terms
        self.sed_mode = sed_mode
        self.band_emb_dim = BAND_EMB_DIM
        self.accumulate = accumulate
        self.include_redshift = include_redshift
        self.tanhshrink_centroids = tanhshrink_centroids
        self.autoscale_outputs = autoscale_outputs
        # DF = 3 → (w_raw, μ_raw, log σ)
        self.multiple_decouple = isinstance(decouple_dim, list)
        DF = 4 # (mu_offset, w_raw, mu_raw, log_sigma)
        if self.multiple_decouple: # (mu_offset, w_offset, w_raw, mu_raw, log_sigma)
            DF += 1
        if self.include_redshift: # (mu_offset, w_offset, redshift, w_raw, mu_raw, log_sigma)
            DF += 1
        if decoupled:
            self.to_params = GeneralizedDecoupledBandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                decouple_dim=decouple_dim,
                band_dot_prod=band_dot_prod,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        else:
            self.to_params = BandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                band_dot_prod=band_dot_prod,
                norm=norm,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        if self.suppress_higher_terms:
            self.supression_parameters = nn.Parameter(torch.randn(size=(1, 1, num_basis)), requires_grad=True)
            self.suppressor_activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
            #self.suppressor = SlidingBasisSuppression(num_basis=num_basis)

        if self.sed_mode:
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=False)
            self.softmax = nn.Softmax(dim=-1)

        if self.tanhshrink_centroids:
            self.tanhshrink = nn.Tanhshrink()
        if self.autoscale_outputs:
            self.scale = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.scale = 1.0

    # --------------------------------------------------------------------- #
    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor, return_parameters: bool=False):
        """
        Args
        ----
        z        : (B, latent_dim)
        t        : (B, T, 1)
        band_idx : (B, T, 1)

        Returns
        -------
        out : (B, T, 1)  scaled flux
        """
        # (B, T, K, 3)  last dim = (μ_raw, w_raw, log σ)
        params = self.to_params(z, t, band_idx)
        if self.multiple_decouple:
            if self.include_redshift:
                mu_offset, w_offset, redshift, w_raw, mu_raw, log_sigma = params.unbind(-1)
                redshift = torch.mean(redshift, dim=-1, keepdim=True)  # Ensemble Average
            else:
                mu_offset, w_offset, w_raw, mu_raw, log_sigma = params.unbind(-1)
            w_offset = torch.mean(w_offset, dim=-1, keepdim=True) # ensemble average
            if self.sed_mode:
                w_offset = torch.mean(w_offset, dim=-2, keepdim=True) # The amplitude is global over all bands
                # NOTE: Should put some consideration into handling mu_offset since offsets might be a little different per band
            w_offset = self.softplus(w_offset) # strictly positive, should consider sacling bounds to data
            if self.accumulate: # want these to end up sorted so that the first weight it positive for the fisrt mu being smallest
                #w_raw = self.sigmoid(w_raw) # All positive
                w_raw = w_raw * w_raw # All positive, weight decay towards zero
                w_raw = w_raw.cumsum(-1) # now they are sorted (Smallest weight first)
                w_raw = self.sigmoid(-w_raw) # Set relative scale, range 1 NOTE: Might set to range of 2 (for [-1, 1]), largest value first
            else:
                w_raw = self.sigmoid(w_raw)  # Set relative scale, range 1 NOTE: Might set to range of 2 (for [-1, 1])
            w_raw = w_raw - torch.mean(w_raw, dim=-1, keepdim=True) # zero-mean in each basis (But not each band)
            w_raw = w_raw * w_offset # Set global scale
        else:
            mu_offset, mu_raw, w_raw, log_sigma = params.unbind(-1)  # each (B, T, K)

        mu_offset = torch.mean(mu_offset, dim=-1, keepdim=True) # 1 centroid per band
        if self.sed_mode:
            mu_offset = torch.mean(mu_offset, dim=-2, keepdim=True) # The center is a global attribute over all bands
            # WARNING: Forcing this over all bands will only allow sed components to adjust their spread, not their offsets!\
            # Empirically, this seems to make the outputs more squiggly 
            # This is probably fine if components on one side cancel themselves out to some degree
            # Should probably think of a better centering method though (in reference to the mean used below on mu_raw)
            # Options: Range (min-max), median (not differentiable), min, max, Tanhshrink, LayerNorm
        if self.accumulate:
            mu_raw = mu_raw * mu_raw # All positive, weight decay towards zero
            mu_raw = mu_raw.cumsum(-1) # sorted with smallest value first
        if self.tanhshrink_centroids:
            mu_raw = self.tanhshrink(mu_raw) + mu_offset # Makes mu_raw small rather than centering so we can have a bias
        else:
            mu_raw = mu_raw + mu_offset - torch.mean(mu_raw, dim=-1, keepdim=True) # Center the offsets in each basis

        mu = mu_raw
        weights = w_raw

        sigma  = self.softplus(log_sigma) + 1e-4
        if self.sed_mode:
            t_expanded = t[:, :, None, :].expand(-1, -1, self.band_emb_dim, self.num_basis) # (B, T, E, K)
            e_b = self.softmax(self.band_emb(band_idx[:, :, 0]))[..., None] #** 2
            if self.include_redshift: # NOTE: Perhaps we update the band embedding with the redshift? 
                redshift = torch.mean(redshift, dim=-2, keepdim=True) # Redshift is the same for all bands
        else:
            t_expanded = t.expand(-1, -1, self.num_basis) # (B, T, K)
            e_b = None

        if self.include_redshift:
            redshift = self.softplus(redshift) - np.log(2) # Allow for small negative redshift and unbounded positive
            t_expanded = t_expanded / (1 + redshift) # convert time to rest frame
            weights = weights / torch.square(1 + redshift) # Convert luminoisty to rest_frame\
            # NOTE: Should mu spacing account for redshift?  Redshifting t kind of already does this
        basis = torch.sigmoid((t_expanded - mu) * sigma)  # (B, T, K)

        if self.suppress_higher_terms: # This allows the network to learn how many basis functions to include
            weights = weights * self.suppressor_activation(self.softplus(self.supression_parameters).cumsum(-1))
            #weights = weights * self.suppressor()

        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()
        if self.include_redshift:
            self.last_redshift = redshift.detach()
        if self.sed_mode:
            self.last_band_embedding = e_b.detach()

        # Weighted sum of bases → (B, T, 1)
        f = torch.mean(weights * basis, dim=-1, keepdim=True)

        if self.sed_mode: # Do a (normalized) dot-product over the SED at each band location
            f = (e_b * f).sum(dim=2) / (e_b).sum(dim=2)

        # NOTE: This mode will require 'weights' to be very large which can hurt with weight decay
        # There is also some unaccounted for non-linear scale factor at play here, might consider setting the global weight scale here
        # Using logscale_output will also make the output function more "wiggly" and will have a harder time supressing higher-order terms
        if self.logscale_output: # In theory this should work better with redshifts due to amplitude scaling
            f = (torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5) * self.scale**2
        else:
            f = (f - 0.5) * self.scale**2

        result = f

        if return_parameters:
            if self.include_redshift:
                parameters = (basis, weights, mu, sigma, e_b, redshift)
            else:
                parameters = (basis, weights, mu, sigma, e_b, None)
            result = (f, parameters)

        return result
    

@register_module
class DoubleSigmoidBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        decoupled=False, decouple_dim=None,
        nlayers=4,
        band_dot_prod=False,
        logscale_output=True,
        norm=False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        include_redshift: bool = False,
        scale_embedding_grad_by_freq: bool = False,
        generate_spectrum: bool = False
    ):
        super().__init__()
        self.num_basis = num_basis
        self.band_dot_prod = band_dot_prod
        self.nlayers = nlayers
        self.logscale_output = logscale_output
        self.sed_mode = sed_mode
        self.band_emb_dim = BAND_EMB_DIM
        self.include_redshift = include_redshift
        self.generate_spectrum = generate_spectrum
        # DF = 3 → (w_raw, μ_raw, log σ)
        self.multiple_decouple = isinstance(decouple_dim, list)
        DF = 6 # ['amplitude', 'rise_rate', 'decline_rate', 'offset', 'width']
        if self.multiple_decouple: # (mu_offset, w_offset, w_raw, mu_raw, log_sigma)
            DF += 1
        if self.include_redshift: # (mu_offset, w_offset, redshift, w_raw, mu_raw, log_sigma)
            DF += 1

        
        input_dim = latent_dim
        if self.generate_spectrum:
            input_dim += BAND_EMB_DIM
            lambda_grid = torch.linspace(0, 1, BAND_EMB_DIM)+0.1
            self.register_buffer("lambda_grid", lambda_grid)
            assert self.include_redshift, "Cannot use generate spectrum without redshift"
            self.redshift_transform = ResNet(BAND_EMB_DIM * (DF-3) * num_basis+BAND_EMB_DIM, BAND_EMB_DIM * (DF-3) * num_basis, nlayers=self.nlayers, nhid=hidden, activation=activation, dropout=dropout, norm=norm)

        self.dof = DF
        self.to_params = DecoupledParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=BAND_EMB_DIM,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                decouple_dim=decouple_dim,
                band_dot_prod=band_dot_prod,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq)
        #if decoupled:
        #    self.to_params = GeneralizedDecoupledBandParametricDecoder(
        #        latent=latent_dim,
        #        hidden=hidden,
        #        K=num_basis,
        #        NUM_BANDS=NUM_BANDS,
        #        BAND_EMB_DIM=BAND_EMB_DIM,
        #        DF=DF,
        #        activation=activation,
        #        dropout=dropout,
        #        decouple_dim=decouple_dim,
        #        band_dot_prod=band_dot_prod,
        #        norm=norm,
        #        rescale_decoupled_network=rescale_decoupled_network,
        #        sed_mode=sed_mode,
        #        scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
        #    )
        #else:
        #    self.to_params = BandParametricDecoder(
        #        latent=latent_dim,
        #        hidden=hidden,
        #        K=num_basis,
        #        NUM_BANDS=NUM_BANDS,
        #        BAND_EMB_DIM=BAND_EMB_DIM,
        #        DF=DF,
        #        activation=activation,
        #        dropout=dropout,
        #        band_dot_prod=band_dot_prod,
        #        norm=norm,
        #        sed_mode=sed_mode,
        #        scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
        #    )
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.basis = DoubleSigmoid()

        if self.sed_mode:
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=False)
            self.softmax = nn.Softmax(dim=-1)

        # --------------------------------------------------------------------- #
        self.global_parameters = ['global_offset', 'global_amplitude']
        self.local_parameters = ['amplitude', 'rise_rate', 'decline_rate', 'offset', 'width']

        self.set_parameter_names()

    def set_parameter_names(self):

        if self.include_redshift:
            self.parameter_names = self.global_parameters + ['redshift'] + self.local_parameters
        else:
            self.parameter_names =  self.global_parameters + self.local_parameters

    def extract_parameters(self, parameters):

        named_parameters = dict(zip(self.parameter_names, parameters.unbind(-1)))
        return named_parameters
    
    def average_parameter(self, parameter):

        if self.sed_mode:
            parameter = torch.mean(parameter, dim=(-1, -2), keepdim=True)
        else:
            parameter = torch.mean(parameter, dim=(-1,), keepdim=True)
        return parameter
    
    def average_parameters(self, parameters, names):

        for name in names:
            if name in parameters:
                parameters[name] = self.average_parameter(parameters[name])
    
    def center_parameter(self, parameter):

        parameter = parameter - torch.mean(parameter, dim=(-1,), keepdim=True)

        return parameter
    
    def center_parameters(self, parameters, names):

        for name in names:
            if name in parameters:
                parameters[name] = self.average_parameter(parameters[name])

    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor, return_parameters: bool=False):
        """
        Args
        ----
        z        : (B, latent_dim)
        t        : (B, T, 1)
        band_idx : (B, T, 1)

        Returns
        -------
        out : (B, T, 1)  scaled flux
        """
        # (B, T, E, K, N)  last dim = (μ_raw, w_raw, log σ)
        params = self.to_params(z, t, band_idx) # (Batch, Time, BAND_EMB, NUM_BASIS, DF)
        param_shape = params.shape
        params = self.extract_parameters(params)
        self.average_parameters(params, ['redshift', 'global_offset', 'global_amplitude'])
        if self.include_redshift:
            params['redshift'] = self.softplus(params['redshift']) - np.log(2) # Allow for small negative redshift and unbounded positive
        if self.generate_spectrum:
            #import sys
            #print([(key, params[key].shape) for key in self.local_parameters], file=sys.stderr)
            #exit()
            params_flat = torch.cat([params[key].reshape(param_shape[0], param_shape[1], -1) for key in self.local_parameters], dim=-1)
            redshifted_lambda = self.lambda_grid[None, None, :] * (1 + params['redshift'].squeeze(-1))
            redshifted_params = torch.cat([params_flat, redshifted_lambda.expand(-1, param_shape[1], -1)], dim=-1)
            redshifted_params = redshifted_params.reshape(param_shape[0], param_shape[1], -1)
            new_params_flat = self.redshift_transform(redshifted_params)
            #torch.cuda.synchronize()
            params.update(
                dict(
                    zip(
                        self.local_parameters, new_params_flat.reshape(
                            param_shape[0], param_shape[1], self.band_emb_dim, self.num_basis, self.dof-3
                            ).unbind(-1)
                        )
                    )
                )
        params['global_amplitude'] = self.softplus(params['global_amplitude'])

        params['offset'] = self.center_parameter(params['offset']) + params['global_offset']
        params['amplitude'] = self.sigmoid(params['amplitude']) * params['global_amplitude']

        params['rise_rate'] = self.softplus(params['rise_rate'])
        params['decline_rate'] = self.softplus(params['decline_rate'])
        params['width'] = self.softplus(params['width'])

        if self.sed_mode:
            t_expanded = t[:, :, None, :].expand(-1, -1, self.band_emb_dim, self.num_basis) # (B, T, E, K)
            #e_b = self.softmax(self.band_emb(band_idx[:, :, 0]))[..., None] #** 2
            e_b = self.band_emb(band_idx[:, :, 0])[..., None]** 2
        else:
            t_expanded = t.expand(-1, -1, self.num_basis) # (B, T, K)
            e_b = None

        if self.include_redshift:
            #params['redshift'] = self.softplus(params['redshift']) - np.log(2) # Allow for small negative redshift and unbounded positive
            #t_expanded = t_expanded / (1 + params['redshift']) # convert time to rest frame
            params['rise_rate'] /= (1 + params['redshift'])
            params['decline_rate'] /= (1 + params['redshift'])
            params['width'] *=  (1 + params['redshift'])
            params['amplitude'] = params['amplitude'] / torch.square(1 + params['redshift']) # Convert luminoisty to rest_frame\
            # NOTE: Should mu spacing account for redshift?  Redshifting t kind of already does this
        
        #basis = self.basis(t_expanded, params['amplitude'], params['rise_rate'], params['decline_rate'], params['offset'], params['width'])

        # Weighted sum of bases → (B, T, 1)
        #f = torch.mean(basis, dim=-1, keepdim=True)

        #if self.sed_mode: # Do a (normalized) dot-product over the SED at each band location
        #    f = (e_b * f).sum(dim=2) / (e_b).sum(dim=2)

        # NOTE: This mode will require 'weights' to be very large which can hurt with weight decay
        # There is also some unaccounted for non-linear scale factor at play here, might consider setting the global weight scale here
        # Using logscale_output will also make the output function more "wiggly" and will have a harder time supressing higher-order terms
        #if self.logscale_output: # In theory this should work better with redshifts due to amplitude scaling
        #    f = (torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5) 
        #else:
        #    f = (f - 0.5) 

        f = self.generate_function(t_expanded, params, e_b)

        result = f

        if return_parameters:
            result = (f, params)

        return result

    @torch.jit.export
    def generate_function(self, t_expanded, params, e_b):

        basis = self.basis(t_expanded, params['amplitude'], params['rise_rate'], params['decline_rate'], params['offset'], params['width'])

        # Weighted sum of bases → (B, T, 1)
        f = torch.mean(basis, dim=-1, keepdim=True)

        if self.sed_mode: # Do a (normalized) dot-product over the SED at each band location
            f = (e_b * f).sum(dim=2) / (e_b).sum(dim=2)

        # NOTE: This mode will require 'weights' to be very large which can hurt with weight decay
        # There is also some unaccounted for non-linear scale factor at play here, might consider setting the global weight scale here
        # Using logscale_output will also make the output function more "wiggly" and will have a harder time supressing higher-order terms
        if self.logscale_output: # In theory this should work better with redshifts due to amplitude scaling
            f = (torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5) 
        else:
            f = (f - 0.5) 

        return f


@register_module
class SpectrumBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        nlayers=4,
        parameter_n_layers=None,
        logscale_output=True,
        norm=False,
        rescale_decoupled_network: bool = False,
        scale_embedding_grad_by_freq: bool = False,
        norm_first: bool = True,
        band_dropout: float = 0.0,
        full_spectrum: bool = False,
        weight_bases: bool = False,
        separable: bool = False,
        norm_params: bool = False,
        log_amp_center: float = 0.0,
        double_sigmoid: bool = False,
        scale_band_embedding: bool = False,
        softmax_time_evolution: bool = False,
        sine_transform: bool = False,
        use_logspace: bool = False,
        norm_band_embedding: bool = False,
        band_embedding_files = None, # Dict
        embedding_expansion=1,
        time_impulse_first: bool = False,
        embed_freqs: bool = False,
        residual_parameter_input: bool=False,
        use_modular_basis: bool = False,
        wave_min=None,
        wave_max=None,
        **kwargs,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.nlayers = nlayers
        self.logscale_output = logscale_output
        self.band_emb_dim = BAND_EMB_DIM
        self.parameter_n_layers = parameter_n_layers
        self.full_spectrum = full_spectrum
        self.weight_bases = weight_bases
        self.separable = separable
        self.log_amp_center = log_amp_center
        self.double_sigmoid = double_sigmoid
        self.softmax_time_evolution = softmax_time_evolution
        self.num_bands = NUM_BANDS
        self.use_logspace = use_logspace
        self.norm_band_embedding = norm_band_embedding
        self.embedding_expansion = embedding_expansion
        self.use_modular_basis = use_modular_basis

        self.band_embedding_files = band_embedding_files

        if parameter_n_layers is None:
            self.parameter_n_layers = nlayers

        self.dof = 3 # (global_offset, global_amplitude, redshift)
        self.input_dim = latent_dim

        if self.band_embedding_files is not None:
            lambda_grid, filters = map_transmission_filters(self.band_embedding_files, n_points=BAND_EMB_DIM, wave_min=wave_min, wave_max=wave_max)
            band_idx = [BAND_IDX_MAP[key] for key in filters]
            throughput = np.vstack(list(filters.values())).astype(np.float32)
            throughput = torch.log(torch.tensor(throughput[np.argsort(band_idx)], requires_grad=False)).clamp_min(-44.0)
            self.band_emb = nn.Embedding.from_pretrained(throughput, freeze=True)
            lambda_grid = torch.tensor(lambda_grid[None, None, :].astype(np.float32), requires_grad=False)
            self.num_bands = len(filters)
            self.band_emb_dim = len(lambda_grid)
        else:
            if wave_min is None:
                wave_min = 0.1
            if wave_max is None:
                wave_max = 2.0
            lambda_grid = torch.linspace(wave_min, wave_max, BAND_EMB_DIM, requires_grad=False)[None, None, :]*100 # (B, S, W) # WARNING! THIS MUST BE MONOTONICALLY INCREASING!
            #lambda_grid = torch.logspace(np.log10(0.1), np.log10(2.0), BAND_EMB_DIM, requires_grad=False)[None, None, :]*100 # (B, S, W) # WARNING! THIS MUST BE MONOTONICALLY INCREASING!
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=scale_embedding_grad_by_freq)
            init.uniform_(self.band_emb.weight, -0.01, 0.01)
        self.register_buffer("lambda_grid", lambda_grid)
        self.band_embedding_scale = np.sqrt(BAND_EMB_DIM/NUM_BANDS) if scale_band_embedding else 1.0

        parameter_dim = hidden
        basis_dim = hidden
        if rescale_decoupled_network:
            parameter_dim = int(hidden * (self.dof) / latent_dim)
            basis_dim = int(hidden * (latent_dim-self.dof) / latent_dim)


        self.to_params = ResidualNetwork(
            n_features=self.dof,
            n_output=self.dof,
            n_hidden=parameter_dim - (parameter_dim % self.dof),
            n_layers=self.parameter_n_layers,
            activation=activation,
            dropout=dropout,
            norm=norm_params,
            n_heads=self.dof,
            residual_input=residual_parameter_input,
        )

        if self.weight_bases:
            self.basis_weight = nn.Sequential(
                    ResidualNetwork(
                    n_features=latent_dim,
                    n_output=self.num_basis,
                    n_hidden=basis_dim,
                    n_layers=nlayers,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    n_heads=1,
                ), nn.Softmax(-1) if self.use_logspace else nn.LogSoftmax(-1)
            )

        partial_network = lambda n_features, n_output: nn.Sequential(
            nn.Linear(n_features, basis_dim - (basis_dim % self.num_basis)),
            ResidualNetwork(
                n_features=basis_dim - (basis_dim % self.num_basis),
                n_output=n_output,
                n_hidden=basis_dim - (basis_dim % self.num_basis),
                n_layers=nlayers,
                activation=activation,
                dropout=dropout,
                norm=norm,
                n_heads=n_output if not self.full_spectrum else self.num_basis,
                norm_first=norm_first
            )
        )
        if self.full_spectrum:
            self.basis = FullSpectrumBasis(
                partial_network=partial_network,
                latent_dim=latent_dim - self.dof,
                wavelength_dim=self.band_emb_dim,
                n_output=self.num_basis
            )
            self.separable=False

        else:
            self.basis = SpectrumBasis(
                    partial_network, 
                    n_features=latent_dim-self.dof,
                    n_output=self.num_basis,
                    separable=separable,
                    sine_transform=sine_transform,
                    embedding_expansion = embedding_expansion,
                    time_impulse_first=time_impulse_first,
                    embed_freqs=embed_freqs,
                    use_modular_basis=use_modular_basis
                )
        
        self.band_dropout = nn.Dropout(band_dropout) # Used for the band embedding to prevent too much weight in one band
        self.band_softmax = nn.Softmax(dim=2)
        self.log10e = np.log10(np.e)

        # --------------------------------------------------------------------- #
        self.global_parameters = ['global_offset', 'global_amplitude', 'redshift']
        self.local_parameters = []
        self.set_parameter_names()


        #self.basis.compile()
        #self.to_params.compile()
        self.compile()


    def set_parameter_names(self):

        self.parameter_names =  self.global_parameters + self.local_parameters

    def extract_parameters(self, parameters):

        named_parameters = dict(zip(self.parameter_names, parameters.unbind(-1)))
        return named_parameters

    
    def get_params(self, z, redshift=None):

        params = self.to_params(z[..., :self.dof]) # (B, dofx1)
        params = self.extract_parameters(params) # (global_offset, global_amplitude, global_redshift), all are shape batch_size
        if redshift is not None:
            params['redshift'] = redshift
        else:
            params['redshift'] = F.softplus(params['redshift']) - np.log(2) # Allow for small negative redshift and unbounded positive

        if self.use_logspace:
            params['global_amplitude'] = params['global_amplitude'] + self.log_amp_center - torch.log1p(params['redshift'])

        else:
            params['global_amplitude'] = torch.exp(params['global_amplitude']+self.log_amp_center)/torch.square((1 + params['redshift'])) # Strictly positive
        #params['global_amplitude'] /= torch.square((1 + params['redshift'])) # (B)

        return params
    
    
    def get_basis(self, time, wavelength, z, params):

        wave_basis, time_basis = self.basis(time, wavelength, z[..., self.dof:]) # (batch_size, seq_len, wavelength_len, 1 x num_basis)
        # wave_basis = F.sigmoid(wave_basis) # Constrain from 0 to 1
        if self.separable or self.use_modular_basis:
            #time_basis = F.sigmoid(time_basis) # Constrain from 0 to 1
            if self.use_logspace:
                wave_basis = F.log_softmax(wave_basis, dim=-2)

            else:
                wave_basis = F.softmax(wave_basis, dim=-2) # Constrain from 0 to 1 but sum to 1 along wave axis
            
            if self.double_sigmoid: # Amplitude is a number from 0 to 1
                
                if self.softmax_time_evolution:
                    if self.use_logspace:
                        time_sum = time_basis.sum(-1)[..., None]
                        time_basis = np.log(self.num_basis) + F.log_softmax(time_basis, dim=-1) - (F.softplus(time_sum) + F.softplus(-time_sum))
                    else:
                        time_basis = self.num_basis * F.softmax(time_basis, dim=-1)*(F.sigmoid(time_basis.sum(-1)[..., None]) * (1 - F.sigmoid(time_basis.sum(-1)[..., None]))) 
                else:
                    if self.use_logspace:
                        time_basis = -(F.softplus(time_basis) + F.softplus(-time_basis))

                    else:
                        time_basis = F.sigmoid(time_basis) * (1 - F.sigmoid(time_basis))
            elif not self.use_modular_basis:
                if self.softmax_time_evolution:
                    if self.use_logspace:
                        time_basis = np.log(self.num_basis) + F.log_softmax(time_basis, dim=-1) + (time_basis.sum(-1)[..., None]) - F.softplus(time_basis.sum(-1)[..., None])

                    else:
                        time_basis = self.num_basis * F.softmax(time_basis, dim=-1)*F.sigmoid(time_basis.sum(-1)[..., None]) 
                else:
                    if self.use_logspace:
                        time_basis = -F.softplus(-time_basis)

                    else:
                        time_basis = F.sigmoid(time_basis)

            else:
                if self.softmax_time_evolution:
                    if self.use_logspace:
                        time_basis = np.log(self.num_basis) + F.log_softmax(time_basis, dim=-1) + torch.log(time_basis.sum(-1)[..., None])

                    else:
                        time_basis = self.num_basis * F.softmax(time_basis, dim=-1)*time_basis.sum(-1)[..., None]
                else:
                    if self.use_logspace:
                        time_basis = torch.log(time_basis)

                    else:
                        time_basis = time_basis
            
        else:
            if self.use_logspace:
                wave_basis = -F.softplus(-wave_basis)

            else:
                wave_basis = F.sigmoid(wave_basis) # Constrain 0 to 1
        if self.use_logspace:
            wave_basis = wave_basis + params['global_amplitude'][..., None, None, None]
        else:
            wave_basis = wave_basis * params['global_amplitude'][..., None, None, None]
        return wave_basis, time_basis
    
    
    def get_band_embedding(self, band_idx, return_log=False):

        if return_log:

            e_b = self.band_dropout(self.band_emb(band_idx[:, :, 0])/self.band_embedding_scale)[..., None] # (B, T, E, 1)
            if self.norm_band_embedding: # Force band_embedding to integrate to 1, possibly helps prevent overweighting
                x = self.lambda_grid[..., None]
                log_x = torch.log(x)
                e_b_norm = self.logsumexp_trapz(e_b, x=x)[:, :, None] 
                e_b = e_b - e_b_norm

        else:

            e_b = self.band_dropout(torch.exp(self.band_emb(band_idx[:, :, 0])/self.band_embedding_scale))[..., None] # (B, T, E, 1)
            if self.norm_band_embedding:
                e_b_norm = torch.trapezoid(e_b, x=self.lambda_grid[..., None], dim=2).clamp_min(1e-5)[:, :, None]
                e_b = e_b / e_b_norm


        return e_b

    def get_temporal_spatial_grid(self, time, wavelength, params):

        wavelength = wavelength / (1 + params['redshift'])[..., None, None] # (B, S, W)
        time = time / (1 + params['redshift'])[..., None, None] - params['global_offset'][..., None, None] / (1 + params['redshift'])[..., None, None] # (B, S, W) #TODO: Reorder this!  Offset before scale!

        return time, wavelength

    def get_basis_weights(self, z):

        return self.basis_weight(z)[:, None, None, :]

    def logsumexp_trapz(self, y, x):

        dx = torch.log(x[:, :, 1:] - x[:, :, :-1])
        y_minus = y[:, :, :-1]
        y_plus = y[:, :, 1:]
        term = y_minus + F.softplus(y_plus - y_minus) + dx - np.log(2)
        logsum = torch.logsumexp(term, dim=2)
        return logsum



    def apply_band_filters(self, sed_surface, e_b):

        #return (e_b * sed_surface).sum(dim=2, keepdim=True) / (e_b).sum(dim=2, keepdim=True)
        if self.use_logspace: # assumes SED surface and e_b are already in logspace, lambda_grid is linear space
            x = self.lambda_grid[..., None]
            log_x = torch.log(x)
            return (self.logsumexp_trapz(e_b + sed_surface + log_x, x=x) - self.logsumexp_trapz(e_b + log_x, x=x))[:, :, None] 

        else:
            return (torch.trapezoid(e_b * sed_surface * self.lambda_grid[..., None], x=self.lambda_grid[..., None], dim=2) / torch.trapezoid(e_b*self.lambda_grid[..., None], x=self.lambda_grid[..., None], dim=2).clamp_min(1e-5))[:, :, None]

    
    def post_process(self, f):


        f = f.sum(dim=-1)
        if self.logscale_output: # In theory this should work better with redshifts due to amplitude scaling
            #light_curve = (torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5) 
            light_curve = (torch.special.xlog1py(torch.sign(f)*self.log10e, torch.absolute(f)) - 0.5)
        else:
            light_curve = (f - 0.5) 

        return light_curve

    @torch.jit.export
    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor, return_parameters: bool=False, redshift=None):
        """
        Args
        ----
        z        : (B, latent_dim)
        t        : (B, T, 1)
        band_idx : (B, T, 1)

        Returns
        -------
        out : (B, T, 1)  scaled flux
        """
        # (B, T, E, K, N)  last dim = (μ_raw, w_raw, log σ)

        params = self.get_params(z, redshift=redshift)

        time, wavelength = self.get_temporal_spatial_grid(t, self.lambda_grid, params)

        wave_basis, time_basis = self.get_basis(time, wavelength, z, params) # (batch_size, seq_len, wavelength_len, 1 x num_basis)

        e_b = self.get_band_embedding(band_idx, return_log=self.use_logspace) # (B, T, E, 1)
        wave_basis = self.apply_band_filters(wave_basis, e_b) # (B, T, 1, N) or (B, 1, 1, N)

        if not self.separable:
            basis = wave_basis # (B, T, 1, N)
        else:
            if self.use_logspace:
                basis = wave_basis + time_basis
            else:
                basis = wave_basis * time_basis # (B, T, 1, N)

        if self.weight_bases:
            if self.use_logspace:
                basis = torch.logsumexp(basis + self.get_basis_weights(z), dim=-1, keepdim=True)
            else:
                basis = torch.sum(basis * self.get_basis_weights(z), dim=-1, keepdim=True) #(B, T, 1, 1)
        else:
            if self.use_logspace:
                 basis = torch.logsumexp(basis, dim=-1, keepdim=True) - np.log(self.num_basis) # (B, T, 1, 1)
            else:
                basis = torch.mean(basis, dim=-1, keepdim=True) # (B, T, 1, 1)

        if self.use_logspace:
            basis = torch.exp(basis.clamp(min=-20.0, max=20.0))
        f = self.post_process(basis) # (B, T, 1)

        if return_parameters:
            result = (f, params)
        else:
            result = f

        return result



    
@register_module
class UnconstrainedSigmoidBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        decoupled=False, decouple_dim=1,
        nlayers=4,
        band_dot_prod=False,
        logscale_output=True,
        norm=False,
        rescale_decoupled_network: bool = False,
        sed_mode: bool = False,
        include_redshift: bool = False,
        autoscale_outputs: bool = False,
        scale_embedding_grad_by_freq: bool = False
    ):
        super().__init__()
        self.num_basis = num_basis
        self.band_dot_prod = band_dot_prod
        self.nlayers = nlayers
        self.logscale_output = logscale_output
        self.sed_mode = sed_mode
        self.band_emb_dim = BAND_EMB_DIM
        self.include_redshift = include_redshift
        self.autoscale_outputs = autoscale_outputs
        # DF = 3 → (w_raw, μ_raw, log σ)
        self.multiple_decouple = isinstance(decouple_dim, list)
        DF = 4 # (mu_offset, w_raw, mu_raw, log_sigma)
        if self.multiple_decouple: # (mu_offset, w_offset, w_raw, mu_raw, log_sigma)
            DF += 1
        if self.include_redshift: # (mu_offset, w_offset, redshift, w_raw, mu_raw, log_sigma)
            DF += 1
        if decoupled:
            self.to_params = GeneralizedDecoupledBandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                decouple_dim=decouple_dim,
                band_dot_prod=band_dot_prod,
                norm=norm,
                rescale_decoupled_network=rescale_decoupled_network,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        else:
            self.to_params = BandParametricDecoder(
                latent=latent_dim,
                hidden=hidden,
                K=num_basis,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF,
                activation=activation,
                dropout=dropout,
                band_dot_prod=band_dot_prod,
                norm=norm,
                sed_mode=sed_mode,
                scale_embedding_grad_by_freq=scale_embedding_grad_by_freq
            )
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()


        if self.sed_mode:
            self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=False)
            self.softmax = nn.Softmax(dim=-1)

        if self.autoscale_outputs:
            self.scale = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.scale = 1.0

    # --------------------------------------------------------------------- #
    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor, return_parameters: bool=False):
        """
        Args
        ----
        z        : (B, latent_dim)
        t        : (B, T, 1)
        band_idx : (B, T, 1)

        Returns
        -------
        out : (B, T, 1)  scaled flux
        """
        # (B, T, K, 3)  last dim = (μ_raw, w_raw, log σ)
        params = self.to_params(z, t, band_idx)
        if self.multiple_decouple:
            if self.include_redshift:
                mu_offset, w_offset, redshift, w_raw, mu_raw, log_sigma = params.unbind(-1)
                redshift = torch.mean(redshift, dim=-1, keepdim=True)  # Ensemble Average
            else:
                mu_offset, w_offset, w_raw, mu_raw, log_sigma = params.unbind(-1)
            w_offset = torch.mean(w_offset, dim=-1, keepdim=True) # ensemble average
            if self.sed_mode:
                w_offset = torch.mean(w_offset, dim=-2, keepdim=True) # The amplitude is global over all bands
                # NOTE: Should put some consideration into handling mu_offset since offsets might be a little different per band
            w_raw = w_raw - w_offset  # Set relative scale, range 1 NOTE: Might set to range of 2 (for [-1, 1])
        else:
            mu_offset, mu_raw, w_raw, log_sigma = params.unbind(-1)  # each (B, T, K)

        mu_offset = torch.mean(mu_offset, dim=-1, keepdim=True) # 1 centroid per band
        if self.sed_mode:
            mu_offset = torch.mean(mu_offset, dim=-2, keepdim=True) # The center is a global attribute over all bands
            # WARNING: Forcing this over all bands will only allow sed components to adjust their spread, not their offsets!\
            # Empirically, this seems to make the outputs more squiggly 
            # This is probably fine if components on one side cancel themselves out to some degree
            # Should probably think of a better centering method though (in reference to the mean used below on mu_raw)
            # Options: Range (min-max), median (not differentiable), min, max, Tanhshrink, LayerNorm

        mu_raw = mu_raw - mu_offset # Center the offsets in each basis

        mu = mu_raw
        weights = w_raw

        sigma  = self.softplus(log_sigma)
        if self.sed_mode:
            t_expanded = t[:, :, None, :].expand(-1, -1, self.band_emb_dim, self.num_basis) # (B, T, E, K)
            e_b = self.softmax(self.band_emb(band_idx[:, :, 0]))[..., None] #** 2
            if self.include_redshift: # NOTE: Perhaps we update the band embedding with the redshift? 
                redshift = torch.mean(redshift, dim=-2, keepdim=True) # Redshift is the same for all bands
        else:
            t_expanded = t.expand(-1, -1, self.num_basis) # (B, T, K)
            e_b = None

        if self.include_redshift:
            redshift = self.softplus(redshift) - np.log(2) # Allow for small negative redshift and unbounded positive
            t_expanded = t_expanded / (1 + redshift) # convert time to rest frame
            weights = weights / torch.square(1 + redshift) # Convert luminoisty to rest_frame\
            # NOTE: Should mu spacing account for redshift?  Redshifting t kind of already does this
        basis = torch.sigmoid((t_expanded - mu) * sigma)  # (B, T, K)


        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()
        if self.include_redshift:
            self.last_redshift = redshift.detach()
        if self.sed_mode:
            self.last_band_embedding = e_b.detach()

        # Weighted sum of bases → (B, T, 1)
        f = torch.mean(weights * basis, dim=-1, keepdim=True)

        if self.sed_mode: # Do a (normalized) dot-product over the SED at each band location
            f = (e_b * f).sum(dim=2) / (e_b).sum(dim=2)

        # NOTE: This mode will require 'weights' to be very large which can hurt with weight decay
        # There is also some unaccounted for non-linear scale factor at play here, might consider setting the global weight scale here
        # Using logscale_output will also make the output function more "wiggly" and will have a harder time supressing higher-order terms
        if self.logscale_output: # In theory this should work better with redshifts due to amplitude scaling
            f = (torch.sign(f)*torch.log10(1+torch.absolute(f)) - 0.5) * self.scale**2
        else:
            f = (f - 0.5) * self.scale**2

        result = f

        if return_parameters:
            if self.include_redshift:
                parameters = (basis, weights, mu, sigma, e_b, redshift)
            else:
                parameters = (basis, weights, mu, sigma, e_b, None)
            result = (f, parameters)

        return result



@register_module
class PiecewiseBandDecoder(nn.Module):
    """
    Band aware piece‑wise light-curve model (Villar+19 Eq. 1).
    Predicts 7 parameters:  A, β, t0, log τ_r, log τ_f, γ, c  per band.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden: int = 128,
        NUM_BANDS: int = 6,
        BAND_EMB_DIM: int = 4,
        activation: str = "GELU",
        dropout: float = 0.0
    ):
        super().__init__()
        self.to_params = BandParametricDecoder(
            latent=latent_dim,
            hidden=hidden,
            K=1,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=7,
            activation=activation,
            dropout=dropout
        )
        self.softplus = nn.Softplus()

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        z         (B, L)
        t         (B, T, 1)
        band_idx  (B, T, 1)
        returns   (B, T, 1)
        """
        # -------- predict parameters ------------------------------------------
        p = self.to_params(z, t, band_idx)  # (B,T,1,7)
        p = p.squeeze(2)  # (B,T,7)

        t0_raw, A_raw, beta, log_tr, log_tf, gamma_raw, c = p.unbind(-1)

        # -------- re‑parameterise ---------------------------------------------
        A = self.softplus(A_raw) + 1e-3
        t0 = t0_raw.tanh()
        τ_r = 0.01 + self.softplus(log_tr)
        τ_f = 0.01 + self.softplus(log_tf)
        γ = self.softplus(gamma_raw)
        t1 = t0 + γ

        # -------- piece‑wise flux ---------------------------------------------
        t_flat = t[:, :, 0]  # explicit slice (B,T)
        t_shift = t_flat - t0
        σ = 1.0 + torch.exp(-t_shift / τ_r)

        F_rise = (A + beta * t_shift) / σ + c
        F_fall = (A + beta * γ) * torch.exp(-(t_flat - t1) / τ_f) / σ + c

        mask = (t_flat < t1).float()  # no ellipsis
        flux = mask * F_rise + (1.0 - mask) * F_fall
        return flux.unsqueeze(-1)



# ---------------------------------------------------------------------------
#  Gumbel-CDF basis decoder  (with dropout regularisation)
# ---------------------------------------------------------------------------
@register_module
class GumbelCDFBasisDecoder(nn.Module):
    """
    Latent-conditioned weighted sum of Gumbel CDF bases, with dropout inside the
    parameter MLP and just before the final projection.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden: int = 128,
        num_basis: int = 8,
        NUM_BANDS: int = 6,
        BAND_EMB_DIM: int = 4,
        activation: str = "GELU",
        dropout: float = 0.10,          # default 10 %
    ) -> None:
        super().__init__()
        self.num_basis = num_basis

        # DF = 3 → (μ_raw, w_raw, log β)
        self.to_params = BandParametricDecoder(
            latent=latent_dim,
            hidden=hidden,
            K=num_basis,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=3,
            activation=activation,
            dropout=dropout,            # dropout inside MLP layers
        )

        # extra dropout between MLP output and basis computation
        self.post_dropout = nn.Dropout(p=dropout)

        self.softplus = nn.Softplus()

    # ---------------------------------------------------------------------
    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor):
        """
        z  : (B, latent_dim)
        t  : (B, T, 1)
        band_idx : (B, T, 1)
        returns   (B, T, 1)
        """
        # (B, T, K, 3) with (μ_raw, w_raw, log β)
        params = self.to_params(z, t, band_idx)
        params = self.post_dropout(params)           # ← new dropout layer

        mu_raw, w_raw, log_beta = params.unbind(-1)

        μ = mu_raw                                   # centre
        β = self.softplus(log_beta) + 1e-4           # positive scale
        w = w_raw                                    # weight (signed)

        # expand time to K bases
        t_exp = t.expand(-1, -1, self.num_basis)

        # Gumbel CDF basis: G = exp(-exp(-(t-μ)/β))
        basis = torch.exp(-torch.exp(-(t_exp - μ) / β))

        # optional debug hooks
        self.last_basis  = basis.detach()
        self.last_weight = w.detach()

        raw = torch.sum(w * basis, dim=-1, keepdim=True)
        out = self.softplus(raw)
        return out
