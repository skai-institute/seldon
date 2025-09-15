# Encoder Walkthrough

**Goal:** Map irregular, multi-band supernova light-curve sequences into a fixed-dimension latent vector `z` suitable for downstream decoding and prediction.

---

## Encoder: `RecurrentODEEncoder` (band-aware, multiband-capable)

```python
NUM_BANDS    = 6   # total photometric bands
BAND_EMB_DIM = 2   # size of band embedding vectors

@register_module
class RecurrentODEEncoder(nn.Module):
    """
    GRU-ODE encoder for irregular, possibly multiband light curves.
    Input:  sequences of [time, normalized flux, ...] plus band indices
    Output: fixed-length latent vector z, shape (B, H)
    """
def __init__(
    self,
    input_dim=3, hidden_dim=128, dropout=0.0,
    BAND_EMB_DIM=2, NUM_BANDS=6,
    t_max=2.0, N_points=50,
    nlayers=4, pooling="mean", norm=True
):
    super().__init__()

    # ---- save config ----
    self.hidden_dim   = hidden_dim        # H = hidden dimension of hidden state
    self.NUM_BANDS    = NUM_BANDS
    self.BAND_EMB_DIM = BAND_EMB_DIM      # Eb = band embedding dimension
    self.t_max        = t_max             # max time for latent rollout
    self.N_points     = N_points          # S = # points in uniform grid
    self.nlayers      = nlayers

    # Band embedding: each band gets its own trainable vector (NUM_BANDS x Eb)
    # scale_grad_by_freq=False means every band (frequent or rare) gets the same gradient scale.
    self.band_emb = nn.Embedding(NUM_BANDS, BAND_EMB_DIM, scale_grad_by_freq=False)

    # Flux embedding: map scalar flux -> richer input_dim-dimensional vector
    #Lifting scalar flux into a nonlinear embedding helps the GRU-ODE learn richer transformations.
    self.float_embedding = FPEmbedding(1, input_dim)

    # Continuous-time encoder: ODE + GRU corrections at observations (from the ODE helpers functions in neuralODE.py)
    self.rnn = RNNODEEncoder(
        hidden_dim,
        batch_first=True,
        bias=True,
        nlayers=nlayers,
        norm=norm
    )

    # Combine flux_emb and band_emb, then project to hidden_dim
    self.linear_layer = nn.Linear(input_dim + BAND_EMB_DIM, hidden_dim)

    # Decoder: integrate hidden forward on a fixed uniform grid
    self.to_latent_timeseries = ODEDecoder(
        hidden_size=hidden_dim,
        activation='Softplus',
        batch_first=True
    )

    # Important: use the SAME integrator as the encoder, for consistency
    self.to_latent_timeseries.integrator = self.rnn.integrator

    # After rollout, concatenate time and pool into fixed-size z
    self.deepset = SimpleDeepSet(
        hidden_dim + 1, hidden_dim,
        activation='PReLU',
        dropout=dropout,
        pooling=pooling
    )
    self.t_grid = None

    # Optional: use PyTorch 2.x compilation for speed
    self.compile(mode='max-autotune-no-cudagraphs')

def forward(self, x, band_idx, mask, time_sorted=None, sequence_batch_mask=None):
    """
    Args:
        x    : (B, T, input_dim)   raw sequence [time, flux, ...]
        band_idx : (B, T, 1)       band indices
        mask : (B, T)              boolean mask (True=observed, False=padded)

    Returns:
        z : (B, hidden_dim) latent representation
    """
    if self.t_grid is None:
        self.t_grid = torch.linspace(
            0.0, self.t_max, self.N_points, device=x.device
        ).detach()  # (S,)
    e_b = self.band_emb(band_idx.squeeze(2))   # (B, T, Eb)
    t   = x[..., 0]                            # (B, T)
    f   = x[..., 1:2]                          # (B, T, 1)

    f_emb = self.float_embedding(f)            # (B, T, input_dim)
    x     = f_emb
    x     = torch.cat([x, e_b], dim=-1)        # (B, T, input_dim+Eb)
    x     = self.linear_layer(x)               # (B, T, H)

    # ensure that padded timesteps or missing data do not contribute when fed into the RNN–ODE encoder.
    x     = x.masked_fill_(~mask.unsqueeze(-1), 0.0)
    _, h_n = self.rnn(x, t, mask, time_sorted, sequence_batch_mask)
    # h_n: (B, H) final hidden representation after ODE+GRU updates
    _, zs = self.to_latent_timeseries(self.t_grid, h_n)  # (B, S, H)

    # Concatenate time to each latent vector
    zs = torch.cat(
        [zs, self.t_grid.expand(zs.shape[0], -1)[..., None]],
        dim=-1
    )  # (B, S, H+1)
    z = self.deepset(zs)  # (B, H)
    return z
