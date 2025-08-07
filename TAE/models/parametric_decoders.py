import numpy as np
import torch
from torch import nn
from TAE.util import register_module
import torch
from TAE.models.ModelNN import ResNet
from torch.nn import functional as F
from TAE.models.ModelNN import ExponentialDistriubtionQuantileMapping, NormalDistriubtionQuantileMapping
import typing

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
        activation: str = "GELU",
        dropout: float = 0.0,
        decouple_dim: typing.Optional[typing.Union[typing.List, int, None]] = None,
        band_dot_prod: bool = False,
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
                band_dot_prod=band_dot_prod)
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
                band_dot_prod=band_dot_prod)
        
        else:
            print(self.decouple_dim)
            self.inp_dim = self.decouple_dim[0]+self.decouple_dim[1]# gives back 2 decoupled slices on the first dim
            self.first_decoder = DecoupledBandParametricDecoder( 
                latent=self.inp_dim,
                hidden=hidden,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=2,
                activation=activation,
                dropout=dropout,
                decouple_dim=self.decouple_dim[0],
                band_dot_prod=band_dot_prod)
            self.decoder = GeneralizedDecoupledBandParametricDecoder(
                latent=latent - self.inp_dim,
                hidden=hidden,
                K=K,
                NUM_BANDS=NUM_BANDS,
                BAND_EMB_DIM=BAND_EMB_DIM,
                DF=DF-2,
                activation=activation,
                dropout=dropout,
                decouple_dim=self.decouple_dim[2:],
                band_dot_prod=band_dot_prod
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
        band_dot_prod: bool = False,
    ):
        super().__init__()
        self.decouple_dim = decouple_dim
        self.band_dot_prod = band_dot_prod

        self.to_time = BandParametricDecoder(
            latent=self.decouple_dim,
            hidden=hidden,
            K=K,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=1,
            activation=activation,
            dropout=dropout,
            band_dot_prod=band_dot_prod)
        self.to_params = BandParametricDecoder(
            latent=latent-self.decouple_dim,
            hidden=hidden,
            K=K,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=DF-1,
            activation=activation,
            dropout=dropout,
            band_dot_prod=band_dot_prod
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
        band_dot_prod = False,
    ):
        super().__init__()
        self.K = K
        self.NUM_BANDS = NUM_BANDS
        self.BAND_EMB_DIM = BAND_EMB_DIM
        self.DF = DF  # the number of parameters each sigmoidal basis function has
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.band_dot_prod = band_dot_prod
        if self.BAND_EMB_DIM > 0:
            self.band_emb = nn.Embedding(self.NUM_BANDS, self.BAND_EMB_DIM, scale_grad_by_freq=True)

        if self.band_dot_prod:
            assert self.BAND_EMB_DIM > 0, "Cannot perform a dot product without a band embedding"
            inp = latent # multiply and sum z @ e_b
            self.mlp = ResNet(inp, BAND_EMB_DIM * self.DF * K, nlayers=4, nhid=hidden, activation=activation, dropout=dropout)
        else:
            inp = latent + BAND_EMB_DIM  # concat z‖e_b
            self.mlp = ResNet(inp, NUM_BANDS * self.DF * K, nlayers=4, nhid=hidden, activation=activation, dropout=dropout)
        #self.mlp = nn.Sequential(
        #    nn.Linear(inp, hidden),
        #    self.activation(),
        #    nn.Dropout(dropout),
        #    nn.Linear(hidden, hidden),
        #    self.activation(),
        #    nn.Dropout(dropout),
        #    nn.Linear(hidden, NUM_BANDS * self.DF * K),  # [a|c|s] for all bases
        #)

    def forward(self, z, t, band_idx):  # t:(B,T,1)

        #z = z.unsqueeze(1).expand(z.shape[0], band_idx.shape[1], z.shape[1]) # (B, S, E)
        if self.BAND_EMB_DIM > 0:
            z = z.unsqueeze(1).expand(z.shape[0], band_idx.shape[1], z.shape[1])
            e_b = self.band_emb(band_idx[:, :, 0])  # (B,seq,E)
            if not self.band_dot_prod:
                z = torch.cat([z, e_b], dim=-1)
            params = self.mlp(z) # (B, 3K) # Should be (B, band, 3k)
        else:
            if not self.band_dot_prod:
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], band_idx.shape[1], -1)
            else:
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], self.BAND_EMB_DIM, -1)
        
        # import pdb; pdb.set_trace()
        #params = self.mlp(z) # (B, 3K) # Should be (B, band, 3k)
        
        if not self.band_dot_prod:
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
            e_b = e_b.unsqueeze(-1).expand(-1, -1, -1, self.DF * self.K) # (b, S, E, DF x K)
            params = params.view(
                params.shape[0], params.shape[1], self.BAND_EMB_DIM, self.DF * self.K
            )  # (B, S ,E , DF x K)
            params = (params * e_b).sum(2) # (B, S, DF x K)
            params = params.view(params.shape[0], -1, self.K, self.DF) # (B, S, K, 3)

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
    ):
        super().__init__()
        self.band_dot_prod = band_dot_prod
        self.num_basis = num_basis
        self.multiple_decouple = isinstance(decouple_dim, list)
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
                band_dot_prod=band_dot_prod
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
                band_dot_prod=band_dot_prod
            )
        self.softplus = nn.Softplus()
        self.exp_map = ExponentialDistriubtionQuantileMapping(init=0.25)
        self.normal_map = NormalDistriubtionQuantileMapping(sigma_init=0.1, mu_init=0.0)

        if self.multiple_decouple:
            self.exp_map_offset = ExponentialDistriubtionQuantileMapping(init=8.0)
            self.normal_map_offset = NormalDistriubtionQuantileMapping(sigma_init=0.1, mu_init=2.0)
        

    def forward(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        t: torch.Tensor,  # (B, T, 1)
        band_idx: torch.Tensor,  # (B, T, 1)
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
        t_expanded = t.expand(-1, -1, self.num_basis)  # (B, T, K)

        basis = torch.exp(-(((t_expanded - mu) * sigma).square()).clamp(min=-20.0, max=20.0))
        #weights = w_raw - torch.mean(w_raw, dim=-1, keepdim=True)  # (B, T, K)
        
        #weights = w_raw.square()

        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()

        # Weighted sum of bases → (B, T, 1)
        out = torch.log(1+torch.mean(weights * basis, dim=-1, keepdim=True)) - 0.5
        return out

@register_module
class SigmoidBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, 
        hidden=128, num_basis=8, 
        NUM_BANDS=6, BAND_EMB_DIM=4, 
        activation="GELU", dropout=0.0, 
        decoupled=False, decouple_dim=1,
        band_dot_prod=False,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.band_dot_prod = band_dot_prod
        # DF = 3 → (w_raw, μ_raw, log σ)
        self.multiple_decouple = isinstance(decouple_dim, list)
        DF = 4
        if self.multiple_decouple:
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
                band_dot_prod=band_dot_prod
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
                band_dot_prod=band_dot_prod
            )
        self.softplus = nn.Softplus()

    # --------------------------------------------------------------------- #
    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx: torch.Tensor):
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
            mu_offset, w_offset, w_raw, mu_raw, log_sigma = params.unbind(-1)
            w_raw = w_raw + w_offset - torch.mean(w_raw, dim=-1, keepdim=True)
        else:
            mu_offset, mu_raw, w_raw, log_sigma = params.unbind(-1)  # each (B, T, K)
        #mu_offset, mu_raw, w_raw, log_sigma = params.unbind(-1)  # each (B, T, K)
        mu_offset = torch.mean(mu_offset, dim=-1, keepdim=True)
        mu_raw = mu_raw + mu_offset - torch.mean(mu_raw, dim=-1, keepdim=True)

        # Centre positions lie in [0, t_max] per light curve
        #t_max = t.max(dim=1, keepdim=True).values  # (B,1,1)
        mu = mu_raw
        #mu = mu_raw.clone() #) * 2 - 1# * 2*t_max - t_max  # (B, T, K)
        #mu[:, :, 1:] = self.softplus(mu_raw[:, :, 1:]) # Cumulative summing the offsets to turn them into deltas
        #mu = torch.cumsum(mu, dim=2)

        sigma = self.softplus(log_sigma) #+ 1e-4  # positive, (B, T, K)
        #sigma = log_sigma
        #sigma = torch.exp(log_sigma.clamp(max=20.0))

        mu     = mu_raw
        sigma  = self.softplus(log_sigma) + 1e-4
        t_expanded = t.expand(-1, -1, self.num_basis)
        basis = torch.sigmoid((t_expanded - mu) * sigma)  # (B, T, K)
        weights = w_raw - torch.mean(w_raw, dim=-1, keepdim=True)  # (B, T, K)

        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()

        # Weighted sum of bases → (B, T, 1)
        out = torch.log(1 + torch.mean(weights * basis, dim=-1, keepdim=True)) - 0.5
        return out


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
