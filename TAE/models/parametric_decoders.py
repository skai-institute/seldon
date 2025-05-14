import numpy as np
import torch
from torch import nn


class BandParametricDecoder(nn.Module):
    def __init__(self, latent: int, hidden: int = 128, K: int = 8, NUM_BANDS=6, BAND_EMB_DIM=4, DF = 3):
        super().__init__()
        self.K        = K
        self.NUM_BANDS = NUM_BANDS
        self.BAND_EMB_DIM = BAND_EMB_DIM
        self.DF = DF
        self.band_emb = nn.Embedding(self.NUM_BANDS, self.BAND_EMB_DIM)

        inp = latent + BAND_EMB_DIM                       # concat z‖e_b
        self.mlp = nn.Sequential(
            nn.Linear(inp, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, NUM_BANDS * self.DF * K)                      # [a|c|s] for all bases
        )

  
        nn.init.uniform_(self.mlp[-1].weight, -0.2, 0.2)

    def forward(self, z, t, band_idx):          # t:(B,T,1)
        e_b   = self.band_emb(band_idx[:, :, 0])        # (B,seq,E)
        z = z.unsqueeze(1).expand(z.shape[0], e_b.shape[1], z.shape[1])
        #import pdb; pdb.set_trace()
        params = self.mlp(torch.cat([z, e_b], dim=-1))  # (B, 3K) # Should be (B, band, 3k)
        params = params.view(params.shape[0], params.shape[1], self.NUM_BANDS, self.DF*self.K)             # (B,band,3K)
        params = torch.gather(params, 2, band_idx.unsqueeze(2).expand(-1, -1, 1, self.DF*self.K)).squeeze(2).view(params.shape[0], -1, self.K, self.DF) # (B, K, 3)
    
        return params # (B, S, K, DF)




class BazinDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden = 128, NUM_BANDS=6, BAND_EMB_DIM=4):
        super().__init__()
        self.hidden = hidden
        self.to_params = BandParametricDecoder(latent_dim, hidden, 1, NUM_BANDS, BAND_EMB_DIM, 4)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, t: torch.Tensor, band_idx):
        #A_raw, t0_raw, log_tr, log_tf = self.to_params(z, t, band_idx).unbind(-1)
        params = self.to_params(z, t, band_idx)
        A_raw = params[:,:,0,0]
        t0_raw = params[:,:,0,1]
        log_tr = params[:,:,0,2]
        log_tf = params[:,:,0,3]
                
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
    def __init__(self, latent_dim: int = 128, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis
        self.to_params = nn.Linear(latent_dim, 3 * num_basis)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor, t: torch.Tensor):  # z: (B,L), t: (B,T,1)
        B, T = t.shape[:2]
        params = self.to_params(z)  # (B, 3K)
        w, mu_raw, log_sigma = torch.chunk(params, 3, dim=-1)  # each (B,K)

        t_max = t.max(dim=1, keepdim=True).values  # (B,1,1)
        mu = torch.sigmoid(mu_raw) * t_max.squeeze(-1)  # (B,K)
        sigma = self.softplus(log_sigma) + 1e-4   # (B,K)

        w     = w.unsqueeze(1)         # (B,1,K)
        mu    = mu.unsqueeze(1)        # (B,1,K)
        sigma = sigma.unsqueeze(1)     # (B,1,K)
        t     = t.expand(-1, -1, self.num_basis)  # (B,T,K)

        self.last_basis = torch.sigmoid((t - mu) / sigma)  # (B,T,K)
        self.last_weight = w
        out = torch.sum(w * self.last_basis, dim=-1, keepdim=True)  # (B,T,1)
        return out
