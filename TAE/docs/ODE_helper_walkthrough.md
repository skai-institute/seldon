Big pictures:
We model hidden dynamics via an ODE: $\dot{h}(t)=f_\theta(h(t))$

## ODEFunc: wraps our learnable vector field $f_\theta$ (a small ResNet).
```python
class ODEFunc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # A learnable vector field 
        self.ode_net = ResNet(*args, **kwargs)

    def forward(self, t, y):
        # Signature matches ODE libs: forward(t, y) -> dy/dt.
        # Here your field is autonomous (ignores t); you could include t if desired.
        return self.ode_net(y)  # optional clamp could stabilize exploding fields
```
Input y is the current hidden state; output is the time derivative.
We ignore t (autonomous ODE)



## Solver wrapper: ODESolver (TorchODE)
Integrators / Solvers evolve states in continuous time:

```python
import torchode as tode

class ODESolver(nn.Module):
    def __init__(self, ode_net):
        super().__init__()
        self.ode_net = ode_net                 # an nn.Module with forward(t, y), i.e. dy/dt
        self.term = tode.ODETerm(self.ode_net) # wrap as a TorchODE term because TorchODE requires ODEs to be in a term format. 
        #This just says “here is the RHS of the ODE.”

        # High-order, adaptive explicit solver.
        self.step_method = tode.Tsit5(self.term)


        # This dynamically chooses how big each time step should be. atol=1.0, rtol=1.0 is very loose tolerances (fast but not super accurate).
        # dt_min < 0 allows **backward** integration.
        self.step_size_controller = tode.PIDController(
            atol=1.0, rtol=1.0, pcoeff=0.2, icoeff=0.5, 
            dcoeff=0.0, term=self.term, dt_max=0.5, dt_min=-0.5
        )

        # The adjoint trick saves memory during training by recomputing states in the backward pass instead of storing everything
        self.solver = tode.AutoDiffAdjoint(self.step_method, self.step_size_controller)
        # torch.compile(self.solver) for speed.
        # self.jit_solver = torch.compile(self.solver)

    def forward(self, x, t):
        # x = initial state(s), shape (batch, features).
        # t = times you want the solution at (e.g., [0, 1, 2, 3]).
        # Return:
        #sol.ts: the actual time points (matches t you gave it).
        #sol.ys: the hidden states at those times, shape (batch, len(t), features).

        # TorchODE uses an InitialValueProblem wrapper
        sol = self.solver.solve(tode.InitialValueProblem(y0=x, t_eval=t))
        return sol.ts, sol.ys
```

POTENTIAL PROBLEMS:
Tolerance choice (atol, rtol) are set to be 1.0, which are loose (fast but not accurate)

## Decoding by integrating the hidden: ODEDecoder
This composes ODEFunc + ODESolver to generate a hidden trajectory given an initial hidden state and a time grid.
```python
@register_module
class ODEDecoder(nn.Module):
    def __init__(self, hidden_size, activation='Softplus', dropout=0.0, dt=0.1, batch_first=False, device=None):
        super().__init__()

        self.activation  = getattr(nn, activation)        # currently unused
        self.hidden_size = hidden_size
        self.device      = device
        self.dt          = dt
        # Vector field: maps (hidden_size -> hidden_size). Depth=nlayers=4 by default.
        self.ode_net     = ODEFunc(hidden_size, hidden_size, nhid=hidden_size, nlayers=4, dropout=dropout)
        # Production integrator
        self.integrator  = ODESolver(self.ode_net)
        self.batch_first = batch_first

    def forward(self, time, hx):
        # time: (T,) or (B, T). If (T,), we broadcast across batch.
        # hx: (B, H) initial hidden state at time[0] (or time[-1] if integrating backward)
        ts, h_p = self.integrator(hx, time.expand(hx.shape[0], -1))  # (B, T), (B, T, H)
        return ts, h_p
```

## Encoding irregular sequences: RNNODEEncoder
Between observation times, evolve hidden state continuously with the ODE; at observation times, update via a GRUCell using the current input. That is, per sample

1. Start from $h_{t^{-}}$.
2. Integrate to the next observation time $t$ : $h_t^{\text {pred }}=\Phi\left(h_{t^{-}} ; t^{-} \rightarrow t\right)$.
3. If there is an observation at $t$, apply GRUCell: $h_t=\operatorname{GRU}\left(x_t, h_t^{\text {pred }}\right)$. Otherwise, keep $h_t=h_t^{\text {pred }}$.


```python
@register_module
class RNNODEEncoder(nn.Module):
    def __init__(self, hidden_size, bias=True, device=None, dtype=None, dt=0.0001, batch_first=False, dropout=0.0, nlayers=4, norm=True):
        super().__init__()
        self.hidden_size  = hidden_size
        self.bias         = bias
        self.device       = device
        self.dtype        = dtype
        self.dt           = dt               # small epsilon for duplicate-time handling
        self.batch_first  = batch_first
        self.dropout      = dropout

        # Discrete update at observation times
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Continuous evolution between observations
        self.ode_net    = ODEFunc(hidden_size, hidden_size, nhid=hidden_size, nlayers=nlayers, dropout=dropout, norm=norm)
        self.integrator = ODESolver(self.ode_net)

    def forward(self, x, time, mask, time_sorted=None, sequence_batch_mask=None, hx=None):
        # S = length of the input sequence per batch.
        # x: (S, B, E) inputs (E == hidden_size expected upstream)
        # time: (S, B) timestamps per step and batch; may be non-monotone across S.
        # mask: (S, B) booleans indicating which (step, batch) actually has an observation
        # hx: (B, H) optional initial hidden; zero if None.

        if self.batch_first:
            x    = x.transpose(0, 1)     # (B, S, E) -> (S, B, E)
            time = time.transpose(0, 1)  # (B, S)   -> (S, B)
            mask = mask.transpose(0, 1)  # (B, S)   -> (S, B)

        #S,B,_
        seq_len, batch_size, _ = x.size()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_t_minus_1 = hx      # previous hidden after last update
        h_t         = hx      # current hidden (will be updated)

        # It loops "backwards" over S.

        for t in range(1, seq_len):
            # Grab two consecutive time slices in reversed order:
            t0 = time[seq_len - t]       # (B,)
            t1 = time[seq_len - t - 1]   # (B,)

            # Build a per-batch 2-point time grid.
            # When t1 == t0, subtract a tiny dt to avoid zero-length steps (which can NaN some solvers).
            ts = torch.stack([t0, t1 - self.dt * (t1 == t0)]).t().contiguous().detach()  # (B, 2)

            # Integrate hidden from previous hidden h_{t-1} to the new time:
            # ODESolver returns (ts, ys); we take the last state per batch: ys[:, -1]
            h_p = self.integrator(h_t_minus_1, ts)[1][:, -1]    # (B, H)

            # mask meaning: mask[s, b] is True if, at sequence step s for batch b, you have a real observation; False if it’s padding/missing
            #We add a trailing dimension ([:, None]) so it broadcasts with (B, H) tensors in the next line.
            total_mask = mask[seq_len - t - 1][:, None]         # (B, 1) boolean


            # Two possibilities per batch item:

            #Observed: use the GRU to correct the ODE prediction with the new measurement:
            #h_gru = GRUCell(x_t, h_p) - remember that GRU mixes the previous hidden (h_{t-1}) and the current input (x_t) using learnable gates.
            #Unobserved: keep the ODE prediction
            h_t = self.rnn_cell(x[seq_len - t - 1], h_p) * total_mask + (~total_mask) * h_p


            h_t_minus_1 = h_t

        # This variant returns only the final hidden; you can also collect the full sequence if needed.
        return None, h_t
```

