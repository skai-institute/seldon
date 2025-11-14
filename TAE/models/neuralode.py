import torch
from torch import nn
from TAE.models.ModelNN import ResNet, SimpleHyperMLP
from TAE.util import register_module
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import torchode as tode


class ODESolver(nn.Module):

    def __init__(self, ode_net, atol=1.0, rtol=1.0, pcoeff=0.2, icoeff=0.5, 
            dcoeff=0.0, dt_max=0.5, dt_min=0.0, use_fixed_solver=False):

        super().__init__()

        self.ode_net = ode_net
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
        

    @torch.jit.export
    def forward(self, x, t_eval=None, t_start=None, t_end=None):
        
        if self.use_fixed_solver:
            if t_eval is None:
                dt0 = t_end-t_start
            else:
                dt0 = (t_eval[:, -1] - t_eval[:, 0])

            sol = self.solver.solve(tode.InitialValueProblem(y0=x, t_eval=t_eval, t_start=t_start, t_end=t_end), dt0=dt0)
        else:
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
                 dcoeff=0.0, dt_max=0.5, dt_min=0.0, use_fixed_solver=False):

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
                 dcoeff=dcoeff, dt_max=dt_max, dt_min=dt_min, use_fixed_solver=use_fixed_solver)
        #self.integrator = odeint

    @torch.jit.export
    def forward(self, x, time, mask, time_sorted=None, sequence_batch_mask=None, hx=None):
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
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
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


    
