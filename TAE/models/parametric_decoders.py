import numpy as np
import torch
from torch import nn


class BandParametricDecoder(nn.Module):
    def __init__(
        self,
        latent: int,
        hidden: int = 128,
        K: int = 8,
        NUM_BANDS=6,
        BAND_EMB_DIM=4,
        DF=3,
    ):
        super().__init__()
        self.K = K
        self.NUM_BANDS = NUM_BANDS
        self.BAND_EMB_DIM = BAND_EMB_DIM
        self.DF = DF  # the number of parameters each sigmoidal basis function has
        self.band_emb = nn.Embedding(self.NUM_BANDS, self.BAND_EMB_DIM)

        inp = latent + BAND_EMB_DIM  # concat z‖e_b
        self.mlp = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_BANDS * self.DF * K),  # [a|c|s] for all bases
        )

        nn.init.uniform_(self.mlp[-1].weight, -0.2, 0.2)

    def forward(self, z, t, band_idx):  # t:(B,T,1)
        e_b = self.band_emb(band_idx[:, :, 0])  # (B,seq,E)
        z = z.unsqueeze(1).expand(z.shape[0], e_b.shape[1], z.shape[1])
        # import pdb; pdb.set_trace()
        params = self.mlp(
            torch.cat([z, e_b], dim=-1)
        )  # (B, 3K) # Should be (B, band, 3k)
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

        return params  # (B, S, K, DF)


class BazinDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden=128, NUM_BANDS=6, BAND_EMB_DIM=4):
        super().__init__()
        self.hidden = hidden
        self.to_params = BandParametricDecoder(
            latent_dim, hidden, 1, NUM_BANDS, BAND_EMB_DIM, 4
        )
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx):
        # A_raw, t0_raw, log_tr, log_tf = self.to_params(z, t, band_idx).unbind(-1)
        params = self.to_params(z, t, band_idx)
        A_raw = params[:, :, 0, 0]
        t0_raw = params[:, :, 0, 1]
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


class SigmoidBasisDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, hidden=128, num_basis=8, NUM_BANDS=6, BAND_EMB_DIM=4
    ):
        super().__init__()
        self.num_basis = num_basis
        # DF = 3 → (w_raw, μ_raw, log σ)
        self.to_params = BandParametricDecoder(
            latent=latent_dim,
            hidden=hidden,
            K=num_basis,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=3,
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        z: torch.Tensor,  # (B, latent_dim)
        t: torch.Tensor,  # (B, T, 1)
        band_idx: torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:  # → (B, T, 1)

        # (B, T, K, 3)  → last dim = (w_raw, μ_raw, log σ)
        params = self.to_params(z, t, band_idx)
        w_raw, mu_raw, log_sigma = params.unbind(-1)  # each (B, T, K)

        # Centre positions lie in [0, t_max] per light curve
        t_max = t.max(dim=1, keepdim=True).values  # (B,1,1)
        mu = torch.sigmoid(mu_raw) * t_max  # (B, T, K)
        sigma = self.softplus(log_sigma) + 1e-4  # positive, (B, T, K)

        # Broadcast time to K bases
        t_expanded = t.expand(-1, -1, self.num_basis)  # (B, T, K)

        basis = torch.sigmoid((t_expanded - mu) / sigma)  # (B, T, K)
        weights = w_raw  # (B, T, K)

        # Save for optional inspection/debugging
        self.last_basis = basis.detach()
        self.last_weight = weights.detach()

        # Weighted sum of bases → (B, T, 1)
        out = torch.sum(weights * basis, dim=-1, keepdim=True)
        return out


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
    ):
        super().__init__()
        self.to_params = BandParametricDecoder(
            latent=latent_dim,
            hidden=hidden,
            K=1,
            NUM_BANDS=NUM_BANDS,
            BAND_EMB_DIM=BAND_EMB_DIM,
            DF=7,
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

        A_raw, beta, t0_raw, log_tr, log_tf, gamma_raw, c = p.unbind(-1)

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
