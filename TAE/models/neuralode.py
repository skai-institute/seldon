import torch
from torch import nn
from TAE.models.ModelNN import ResNet, SimpleHyperMLP
from TAE.util import register_module
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import torchode as tode

@register_module
class ExponentialODESolver(nn.Module):
    '''Use a first-order exponential integrator to solve 
    the differential equation

    y' = -A y + f(y)

    where f(y) is a provided function.
    This integrator has the advantage of requiring only
    single evaluations in the RNN loop
    and acts as a residual connection
    on the exponential decay of information
    between successive points'''

    def __init__(self, ode_net, A0=0.0, fixed_solver_steps=1, epsilon=1e-4):

        super().__init__()
        self.ode_net = ode_net
        self.A0 = nn.Parameter(torch.tensor(A0, requires_grad=True))
        self.softplus = nn.Softplus()
        self.fixed_solver_steps = fixed_solver_steps
        self.epsilon = epsilon

    @property
    def A(self):

        return self.softplus(self.A0) + self.epsilon

    def step(self, x0, h):

        coeff = torch.exp(-self.A*h)

        return coeff * x0 + (1 - coeff) / self.A * self.ode_net(x0)

    def solve(self, x, t_eval=None, t_start=None, t_end=None):

        # t_eval is B, S or S
        # t_start and t_end are B
        if t_eval is None:
            uniform_grid = torch.linspace(0, 1.0, self.fixed_solver_steps, device=x.device)
            t_eval = uniform_grid[None, :] * (t_end - t_start)[:, None] + t_start[:, None]
        dts = t_eval[..., 1:] - t_eval[..., -1]
        out_length = t_eval.shape[-1]
        out_tensors = [x]
        for dt in dts.unbind(-1):
            out_tensors.append(self.step(out_tensors[-1], dt))

        return t_eval, torch.stack(out_tensors, dim=-2)



    def forward(self, x, t_eval=None, t_start=None, t_end=None):

        ts, ys = self.solve(x=x, t_eval=t_eval, t_start=t_start, t_end=t_end)

        return ts, ys

@register_module
class ODESolver(nn.Module):

    def __init__(self, ode_net, atol=1e-6, rtol=1e-3, pcoeff=0.2, icoeff=0.5, 
            dcoeff=0.0, dt_max=0.5, dt_min=0.0, use_fixed_solver=False, fixed_solver_steps=1):

        super().__init__()

        self.ode_net = ode_net
        self.fixed_solver_steps = fixed_solver_steps
        self.use_fixed_solver = use_fixed_solver
        self.term = tode.ODETerm(self.ode_net)
        self.step_method = tode.Tsit5(self.term)
        if use_fixed_solver:
            self.step_size_controller = tode.FixedStepController()
        else:
            self.step_size_controller = tode.PIDController(
                atol=atol, rtol=rtol, pcoeff=pcoeff, icoeff=icoeff, 
                dcoeff=dcoeff, term=self.term, dt_max=dt_max, dt_min=dt_min
            )
        
        self.solver = tode.AutoDiffAdjoint(self.step_method, 
                                           self.step_size_controller, 
                                           backprop_through_step_size_control=True)
        #self.solver = tode.BacksolveAdjoint(self.term, self.step_method, self.step_size_controller)
        #self.jit_solver = torch.compile(self.solver)
        self.compile(mode='max-autotune-no-cudagraphs')
        

    @torch.jit.export
    def forward(self, x, t_eval=None, t_start=None, t_end=None):
        
        if self.use_fixed_solver:
            if t_eval is None:
                dt0 = (t_end-t_start)/self.fixed_solver_steps
            else:
                dt0 = (t_eval[:, -1] - t_eval[:, 0])/self.fixed_solver_steps

            sol = self.solver.solve(tode.InitialValueProblem(y0=x, t_eval=t_eval, t_start=t_start, t_end=t_end), dt0=dt0)
        else:
            if t_start is not None and t_end is not None:
                t_end = torch.where((t_start == t_end), t_end + (t_start == t_end) * 1e-3, t_end) # prevents fixed solver from breaking
            sol = self.solver.solve(tode.InitialValueProblem(y0=x, t_eval=t_eval, t_start=t_start, t_end=t_end))

        return sol.ts, sol.ys

@register_module
class ODEDecoder(nn.Module):

    def __init__(self, hidden_size, activation='Softplus', dropout=0.0, dt=0.1, batch_first=False, device=None):

        super().__init__()

        self.activation = getattr(nn, activation)
        self.hidden_size = hidden_size
        self.device=device
        self.dt = dt
        self.ode_net = ODEFunc(hidden_size, hidden_size, nhid=hidden_size, nlayers=4, dropout=dropout)
        self.integrator = ODESolver(self.ode_net)
        #self.integrator = odeint

        self.batch_first = batch_first

    def forward(self, time, hx):

        ts, h_p = self.integrator(hx, time.expand(hx.shape[0], -1))
        return ts, h_p
    
@register_module
class HyperODESolver(nn.Module):

    def __init__(self, hidden_size, hyper_hidden_dim, num_layers=3, activation="Softplus", dt=0.0, batch_first=True, device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.device=device
        self.dt = dt
        self.batch_first = batch_first
        self.hyper_hidden_dim = hyper_hidden_dim
        self.num_layers = num_layers

        self.mlp = SimpleHyperMLP(input_dim=hidden_size, 
                                  output_dim=hidden_size, 
                                  hidden_dim=hidden_size,
                                  hyper_hidden_dim=hyper_hidden_dim,
                                  activation=activation, 
                                  input_dim_cond=1,
                                  num_layers=num_layers)
        
    def forward(self, hx, time):
        # time: (B, S)
        # hx: (B, E)
        #import sys
        #print(time.shape, hx.shape, file=sys.stderr)
        dt = (time[:, 1:] - time[:, :-1])[..., None] # (B, S*, 1), S* is number of time points minus 1
        _, sequence_length, _ = dt.shape
        #print(dt.shape, sequence_length, file=sys.stderr)
        output = [hx]
        h_t_minus_1 = hx
        for i in range(sequence_length):
            hx = h_t_minus_1 + self.mlp(h_t_minus_1, dt[:, i]) # dt[:, i] is (B, 1)
            output.append(hx)
            h_t_minus_1 = hx
        solution = torch.stack(output).transpose(0, 1) # The first output value should always be the input (B, S, E)
        #print(solution.shape, file=sys.stderr)
        return time, solution

@register_module
class RNNODECell(nn.Module):

    def __init__(self, rnn_cell, ode_solver):
        # NOTE: This will have issues when not using the fixed solver as it will not check if integrating from t0=t1

        super().__init__()

        self.rnn_cell = rnn_cell
        self.ode_solver = ode_solver

        self.hidden_size = self.rnn_cell.hidden_size
        self.input_size = self.rnn_cell.input_size

    #@torch.jit.export
    def forward(self, x0, x1, t0, t1, mask0, mask1, hx=None):
        # RNNUpdate and ODE Intgrate from t0 to t1
        # x0, x1: (batch_size x num_features)
        # t0, t1: (batch_size)
        # mask0, mask1: (batch_size)

        batch_size = x0.size(0)
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x0.device)
  
        #hx = torch.where((t0 == t1)[..., None], hx, self.ode_solver(hx, t_start=t0, t_end=t1)[1][..., -1, :])
        hx = torch.where(mask0[..., None], hx, self.ode_solver(hx, t_start=t0, t_end=t1)[1][..., -1, :])
        hx = torch.where(mask1[..., None], self.rnn_cell(x1, hx), hx)

        return hx

@register_module
class PaddedRNNODECell(nn.Module):

    def __init__(self, rnn_cell, ode_solver):
        # NOTE: This will have issues when not using the fixed solver as it will not check if integrating from t0=t1

        super().__init__()

        self.rnn_cell = rnn_cell
        self.ode_solver = ode_solver

        self.hidden_size = self.rnn_cell.hidden_size
        self.input_size = self.rnn_cell.input_size

    #@torch.jit.export
    def forward(self, x0, x1, t0, t1, mask0, mask1, hx=None):
        # RNNUpdate and ODE Intgrate from t0 to t1
        # x0, x1: (batch_size x num_features)
        # t0, t1: (batch_size)
        # mask0, mask1: (batch_size)
        # mask0 is the padding mask
        # mask1 is the skip mask

        batch_size = x0.size(0)
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x0.device)
  
        hx = torch.where(mask0[..., None], hx, self.ode_solver(hx, t_start=t0, t_end=t1)[1][..., -1, :])
        hx = torch.where(mask1[..., None], self.rnn_cell(x1, hx), hx)

        return hx

register_module(nn.GRUCell)
register_module(nn.RNNCell)

@register_module
class LNGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.ln_x = nn.LayerNorm(3 * hidden_size)
        self.ln_h = nn.LayerNorm(3 * hidden_size)

        self.bias = nn.Parameter(torch.zeros(3 * hidden_size, requires_grad=True))

        self.sz = nn.Sigmoid()
        self.sr = nn.Sigmoid()
        self.tanh = nn.Tanh()

    @torch.jit.export
    def forward(self, input, hx):
        x_proj = self.ln_x(self.W(input))
        h_proj = self.ln_h(self.U(hx))

        preact = x_proj + h_proj + self.bias

        z_, r_, n_ = preact.chunk(3, dim=-1)

        z = self.sz(z_)
        r = self.sr(r_)
        n = self.tanh(n_ + r * h_proj[:, 2*self.hidden_size:])

        h_new = (1 - z) * hx + z * n
        return h_new

@register_module
class RNNODEEncoderV2(nn.Module):

    def __init__(self, rnn_ode_cell, use_class_token=False, return_sequence=False):

        super().__init__()
        self.rnn_ode_cell = rnn_ode_cell
        self.integrator = rnn_ode_cell.ode_solver
        self.hidden_size = self.rnn_ode_cell.hidden_size
        self.use_class_token = use_class_token
        self.return_sequence = return_sequence

        if self.use_class_token:
            self.class_token = nn.Parameter(torch.randn(1, self.hidden_size, requires_grad=True))
        else:
            self.class_token = 0

    @torch.jit.export
    def forward(self, x, t, mask, hx=None, reverse=True, pad_mask=None):

        # RNNUpdate and ODE Intgrate from t0 to t1
        # x: (batch_size x sequence_length x num_features)
        # t: (batch_size x sequence_length)
        # mask: (batch_size x sequence_length)

        batch_size, sequence_length, *_ = x.size()
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device) + self.class_token

        if reverse:
            x0, x1, t0, t1, mask0_, mask1 = x[..., 1:, :], x[..., :-1, :], t[..., 1:], t[..., :-1], mask[..., 1:], mask[..., :-1]
            if pad_mask is not None:
                mask0 = ~pad_mask[..., 1:]
            else:
                #mask0 = mask0_
                #mask0 = t0 == t1 # This is an inverted mask where True skips the ode.  Integrating 0 time has numerical stability issues with adaptive solvers
                mask0 = (~torch.cummax(mask0_.flip(dims=(-1,)), dim=-1).values.flip(dims=(-1,))) | (t0 == t1)

        else:
            x0, x1, t0, t1, mask0_, mask1 = x[..., :-1, :], x[..., 1:, :], t[..., :-1], t[..., 1:], mask[..., :-1], mask[..., 1:]
            if pad_mask is not None:
                mask0 = ~pad_mask[..., :-1]
            else:
                #mask0 = mask0_
                #mask0 = t0 == t1
                mask0 = (~torch.cummax(mask0_, dim=-1).values) | (t0 == t1)

        h_t_minus_1 = hx.clone()
        h_t = hx.clone()
        if self.return_sequence:
            outputs = []
        else:
            outputs = None
        i = 0
        if reverse: i = sequence_length - 1 - i
        h_t = torch.where(mask[..., i, None], self.rnn_ode_cell.rnn_cell(x[..., i, :], hx=h_t_minus_1),  h_t_minus_1)
        if self.return_sequence:
            outputs.append(h_t.clone())
        h_t_minus_1 = h_t.clone()

        for i in range(sequence_length - 1):
            if reverse: i = sequence_length - 2 - i
            h_t = self.rnn_ode_cell(x0[..., i, :], x1[..., i, :], t0[..., i], t1[..., i], mask0[..., i], mask1[..., i], hx=h_t_minus_1)
            if self.return_sequence:
                outputs.append(h_t.clone())
            h_t_minus_1 = h_t.clone()

        if self.return_sequence:
            outputs = torch.stack(outputs)
            outputs = outputs.transpose(0, 1).contiguous()

        return outputs, h_t


@register_module
class VariationalODERNN(nn.Module):

    def __init__(self, rnn_ode_cell, autoencoder):

        super().__init__()
        self.rnn_ode_cell = rnn_ode_cell
        self.integrator = rnn_ode_cell.ode_solver
        self.hidden_size = self.rnn_ode_cell.hidden_size
        self.autoencoder = autoencoder

    def collate(self, outputs):

        return {key:torch.stack(outputs[key]).transpose(0, 1).contiguous() for key in outputs}

    @torch.jit.export
    def forward(self, x, t, mask, hx=None, reverse=True):

        # RNNUpdate and ODE Intgrate from t0 to t1
        # x: (batch_size x sequence_length x num_features)
        # t: (batch_size x sequence_length)
        # mask: (batch_size x sequence_length)

        batch_size, sequence_length, *_ = x.size()
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        if reverse:
            x0, x1, t0, t1, mask0, mask1 = x[..., 1:, :], x[..., :-1, :], t[..., 1:], t[..., :-1], mask[..., 1:], mask[..., :-1]
        else:
            x0, x1, t0, t1, mask0, mask1 = x[..., :-1, :], x[..., 1:, :], t[..., :-1], t[..., 1:], mask[..., :-1], mask[..., 1:]

        outputs = []
        i = 0
        if reverse: i = sequence_length - 1 - i
        h_t = self.autoencoder(torch.where(mask[..., i, None], self.rnn_ode_cell.rnn_cell(x[..., i, :], hx=hx),  hx))
        
        outputs.append(h_t)
        h_t_minus_1 = h_t['output'].clone()

        for i in range(sequence_length - 1):
            if reverse: i = sequence_length - 2 - i
            h_t = self.autoencoder(self.rnn_ode_cell(x0[..., i, :], x1[..., i, :], t0[..., i], t1[..., i], mask0[..., i], mask1[..., i], hx=h_t_minus_1))
            outputs.append(h_t)
            h_t_minus_1 = h_t['output'].clone()

        outputs = self.collate(outputs)

        return outputs, h_t_minus_1

@register_module
class RNNODEEncoder(nn.Module):
    
    def __init__(self, hidden_size, 
                 bias=True, 
                 device=None, 
                 dtype=None, 
                 dt=0.0001, 
                 batch_first=False, 
                 dropout=0.0, 
                 nlayers=4, 
                 norm=True,
                 activation='Softplus',
                 use_hyper_integrator=False,
                 atol=1.0, rtol=1.0, pcoeff=0.2, icoeff=0.5, 
                 dcoeff=0.0, dt_max=0.5, dt_min=0.0, use_fixed_solver=False,
                 use_class_token=False, fixed_solver_steps=1):

        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.batch_first = batch_first
        self.dropout = dropout
        self.activation = activation
        self.use_hyper_integrator = use_hyper_integrator
        self.use_fixed_solver = use_fixed_solver
        self.use_class_token = use_class_token
        self.fixed_solver_steps = fixed_solver_steps

        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        if self.use_hyper_integrator:
            self.integrator = HyperODESolver(hidden_size=hidden_size, 
                                             hyper_hidden_dim=4,
                                             device=device, 
                                             activation=activation,
                                             num_layers=nlayers+1)
        else:
            self.ode_net = ODEFunc(hidden_size, 
                                hidden_size, 
                                nhid=hidden_size, 
                                nlayers=nlayers, 
                                dropout=dropout, 
                                norm=norm,
                                activation=activation,
                                bias=bias)
            self.integrator = ODESolver(self.ode_net, atol=atol, rtol=rtol, pcoeff=pcoeff, icoeff=icoeff, 
                 dcoeff=dcoeff, dt_max=dt_max, dt_min=dt_min, use_fixed_solver=use_fixed_solver, fixed_solver_steps=self.fixed_solver_steps)
        #self.integrator = odeint
        if self.use_class_token:
            self.class_token = nn.Parameter(torch.randn(1, self.hidden_size, requires_grad=True))
        else:
            self.class_token = 0

    @torch.jit.export
    def forward(self, x, time, mask, hx=None):
        # x is (S, B, E) where E is input_size
        # time is (S, B)
        # Want (S, B) iteration, mask updates at t = batch[time], pre-sort time
        #assert torch.all(torch.isfinite(x)), "X Emb Again!"
        if self.batch_first:
            x = x.transpose(0, 1)
            time = time.transpose(0, 1)
            mask = mask.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device) + self.class_token
        h_t_minus_1 = hx
        h_t = hx

        # TODO: This function is wrong if there is sparse masked points (or band masks!)
        # The h_p's need to be maintained when evolved to masked points in time
        # or we can forward fill them in the dataloader
        # we should think about which method is more general and easier to extend
        output = []
        for t in range(1, seq_len):
            t0 = time[seq_len-t] # (B)
            t1 = time[seq_len-t-1] # (B)
            time_mask = (t1 == t0)
            #ts = torch.stack([t0, t1-self.dt*time_mask]).t().contiguous()#.detach() # (B, 2) # USE THE t1==t0 MASK!
            #h_p = self.integrator(h_t_minus_1, ts)[1][:, -1]*~time_mask[:, None] + h_t_minus_1 * time_mask[:, None]#* mask[seq_len-t][:, None] # (B, 2), when t1 == t0 this gives nans!
            h_p = self.integrator(h_t_minus_1, None, t_start=t0, t_end=t1-self.dt*time_mask)[1][:, -1]*~time_mask[:, None] + h_t_minus_1 * time_mask[:, None]#* mask[seq_len-t][:, None] # (B, 2), when t1 == t0 this gives nans!

            #assert torch.all(torch.isfinite(h_p)), ts
            total_mask = mask[seq_len-t-1][:, None]
            h_t = self.rnn_cell(x[seq_len-t-1], h_p) * total_mask + (~total_mask) * h_p #h_t_minus_1, # NOTE: THIS IS THE CHANGE
            #assert torch.all(torch.isfinite(h_t))
            output.append(h_t)
            h_t_minus_1 = h_t
            #break
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        #return None, h_t
        return output, h_t
   

@register_module
class ODEModelWrapper(nn.Module):

    '''A wrapper around a neural network to provide an interface to ODE solvers'''

    def __init__(self, model, autonomous=True):
        super().__init__()

        self.model = model
        self.autonomous = autonomous

    def forward(self, t, y):

        if self.autonomous:
            input_data = y
        else:
            input_data = torch.cat([t[..., None], y], dim=-1)

        return self.model(input_data)

class ODEFunc(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.ode_net = ResNet(*args, **kwargs)

    def forward(self, t, y):

        return self.ode_net(y)#.clamp(min=-20.0, max=20.0)



class ForwardEulerIntegratorConst(nn.Module):

    def __init__(self):
        super().__init__()

        ...

    def forward(self, f, f0, t0, t1):
        # f0 is (B, E)
        # t0 is (B)

        #t0 = t0.expand(f0.shape[0], 1)
        #t1 = t1.expand(f0.shape[0], 1)
        dt = (t1 - t0)/10
        #print('dt shape:', dt.shape)
        #print('f0 shape:', f0.shape)
        #print('t0 shape:', t0.shape)
        t = t0
        for i in range(10):
            inp = torch.cat((f0, t.expand(f0.shape[0], 1)), dim=-1)
            f0 = f0 + f(inp) * dt#.expand(f0.shape[0], 1)
            t = t + dt
            
        return f0
    

class ForwardEulerIntegrator(nn.Module):

    def __init__(self):
        super().__init__()

        ...

    def forward(self, f, f0, t0, t1):
        # f0 is (B, E)
        # t0 is (B)

        dt = (t1 - t0)/10.0
        #print('dt shape:', dt.shape)
        #print('f0 shape:', f0.shape)
        #print('t0 shape:', t0.shape)
        t = t0
        for i in range(10):
            inp = torch.cat((f0, t[:, None]), dim=-1)
            f0 = f0 + f(inp) * dt[:, None]
            t = t + dt
            
        return f0


    
