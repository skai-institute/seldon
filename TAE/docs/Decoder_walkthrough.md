# Decoder Walkthrough

**Goal:** Map a latent vector `z` and per‑timestep band indices into `K * DF` parameters per time step that a downstream basis decoder (Gaussian/functional) converts into flux.

---

## Decoder: `BandParametricDecoder` (band‑aware parameter head)

```python
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
    ):
        super().__init__()

        # config
        self.K            = K                 # number of basis functions per band
        self.NUM_BANDS    = NUM_BANDS         # total photometric bands
        self.BAND_EMB_DIM = BAND_EMB_DIM      # Eb = band embedding dim
        self.DF           = DF                # params per basis (e.g., 8)
        self.activation   = getattr(nn, activation)
        self.dropout      = dropout
        self.band_dot_prod= band_dot_prod     #False: concat mode, True: dot‑product mode
        self.nlayers      = nlayers

        # Optional: per‑band learnable embeddings e_b \in R^{Eb}
        if self.BAND_EMB_DIM > 0:
            self.band_emb = nn.Embedding(self.NUM_BANDS, self.BAND_EMB_DIM,
                                         scale_grad_by_freq=False)

        # Two injection modes for band information:
        if self.band_dot_prod:
            assert self.BAND_EMB_DIM > 0, "dot‑product mode requires band embeddings"
            inp = latent                       # z only; band influence via projection later
            out = self.BAND_EMB_DIM * self.DF * K
            self.mlp = ResNet(inp, out,
                              nlayers=self.nlayers, nhid=hidden,
                              activation=activation, dropout=dropout)
        else:
            inp = latent + self.BAND_EMB_DIM   # concat z||e_b
            out = self.NUM_BANDS * self.DF * K # predict all bands then gather one
            self.mlp = ResNet(inp, out,
                              nlayers=self.nlayers, nhid=hidden,
                              activation=activation, dropout=dropout)
```

**What this builds:** a per‑time‑step MLP that outputs either

* **concat mode (`band_dot_prod=False`)**: `[NUM_BANDS, DF * K]` logits so we can gather the requested band, or
* **dot‑product mode (`True`)**: `[BAND_EMB_DIM, DF * K]` logits that get linearly projected by the band embedding.

---

### `forward`

```python
    def forward(self, z, t, band_idx):  # t: (B, S, 1) — not used here
        # z: (B, E)
        # band_idx: (B, S, 1) integers in [0, NUM_BANDS)

        if self.BAND_EMB_DIM > 0:
            # Broadcast z across time so each step sees the same latent.
            z = z.unsqueeze(1).expand(z.shape[0], band_idx.shape[1], z.shape[1])  # (B, S, E)
            # Look up per‑step band embeddings.
            e_b = self.band_emb(band_idx[:, :, 0])  # (B, S, Eb)

            if not self.band_dot_prod:
                # Concat mode: append e_b to z at every timestep.
                z = torch.cat([z, e_b], dim=-1)  # (B, S, E+Eb)
            # Run the MLP per timestep to produce raw parameters.
            params = self.mlp(z)  # (B, S, out)
        else:
            # No embedding available: produce once and broadcast over time.
            if not self.band_dot_prod:
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], band_idx.shape[1], -1)  # (B, S, out)
            else:
                # Dot‑product mode is not meaningful when Eb=0; kept for API symmetry.
                params = self.mlp(z).unsqueeze(1).expand(z.shape[0], self.BAND_EMB_DIM, -1)

        if not self.band_dot_prod:
            # Reshape to separate bands, then gather the requested band at each timestep.
            params = params.view(params.shape[0], params.shape[1], self.NUM_BANDS, self.DF * self.K)  # (B, S, BANDS, DF*K)
            params = (
                torch.gather(
                    params, 2, band_idx.unsqueeze(2).expand(-1, -1, 1, self.DF * self.K)
                )
                .squeeze(2)  # (B, S, DF×K)
                .view(params.shape[0], -1, self.K, self.DF)  # (B, S, K, DF)
            )
        else:
            # Project MLP output in embedding space down to the specific band via <·, e_b>.
            e_b = e_b.unsqueeze(-1).expand(-1, -1, -1, self.DF * self.K)  # (B, S, Eb, DF*K)
            params = params.view(params.shape[0], params.shape[1], self.BAND_EMB_DIM, self.DF * self.K)  # (B, S, Eb, DF*K)
            params = (params * e_b).sum(2)  # inner product over Eb → (B, S, DF*K)
            params = params.view(params.shape[0], -1, self.K, self.DF)  # (B, S, K, DF)

        return params  # (B, S, K, DF)
```

**Notes:**

* `t` is intentionally unused here; time is consumed by the basis decoder when evaluating bases.
* Output shape `(B, S, K, DF)` matches the contract expected by `GaussianBasisDecoder`


## Shapes

* `z`: `(B, E)` -> broadcast to `(B, S, E)`
* `band_idx`: `(B, S, 1)` → drives `e_b` lookup and `gather`
* Output: `(B, S, K, DF)`


## Minimal usage

```python
head = BandParametricDecoder(latent=E, hidden=128, K=8, NUM_BANDS=6,
                             BAND_EMB_DIM=4, DF=4, nlayers=4,
                             band_dot_prod=False)
params = head(z, t, band_idx)  # (B, S, K, DF)
```
