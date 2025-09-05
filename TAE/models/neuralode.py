import torch
from torch import nn
from TAE.models.ModelNN import ResNet
from TAE.util import register_module
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import torchode as tode


class ODESolver(nn.Module):

    def __init__(self, ode_net):

        super().__init__()

        self.ode_net = ode_net
        self.term = tode.ODETerm(self.ode_net)
        #self.step_method = tode.Euler(self.term)
        self.step_method = tode.Tsit5(self.term)
        #self.step_size_controller = tode.IntegralController(atol=1e-5, rtol=1e-3, term=self.term)
        #self.step_size_controller = tode.FixedStepController()
        self.step_size_controller = tode.PIDController(
            atol=1.0, rtol=1.0, pcoeff=0.2, icoeff=0.5, 
            dcoeff=0.0, term=self.term, dt_max=0.5, dt_min=-0.5
        )
        self.solver = tode.AutoDiffAdjoint(self.step_method, self.step_size_controller)
        #self.solver = tode.BacksolveAdjoint(self.term, self.step_method, self.step_size_controller)
        #self.jit_solver = torch.compile(self.solver)

    def forward(self, x, t):
        #dt0 = torch.full((x.shape[0],), 0.1, device=x.device).detach()

        sol = self.solver.solve(tode.InitialValueProblem(y0=x, t_eval=t))
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
class RNNODEEncoder(nn.Module):
    
    def __init__(self, hidden_size, bias=True, device=None, dtype=None, dt=0.0001, batch_first=False, dropout=0.0, nlayers=4, norm=True):

        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.batch_first = batch_first
        self.dropout = dropout

        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.ode_net = ODEFunc(hidden_size, hidden_size, nhid=hidden_size, nlayers=nlayers, dropout=dropout, norm=norm)
        self.integrator = ODESolver(self.ode_net)
        #self.integrator = odeint

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
        #output = []
        for t in range(1, seq_len):
            t0 = time[seq_len-t] # (B)
            t1 = time[seq_len-t-1] # (B)
            ts = torch.stack([t0, t1-self.dt*(t1==t0)]).t().contiguous().detach() # (B, 2) # USE THE t1==t0 MASK!
            h_p = self.integrator(h_t_minus_1, ts)[1][:, -1] #* mask[seq_len-t][:, None] # (B, 2), when t1 == t0 this gives nans!
            #assert torch.all(torch.isfinite(h_p)), ts
            total_mask = mask[seq_len-t-1][:, None]
            h_t = self.rnn_cell(x[seq_len-t-1], h_p) * total_mask + (~total_mask) * h_p #h_t_minus_1, # NOTE: THIS IS THE CHANGE
            #assert torch.all(torch.isfinite(h_t))
            #output.append(h_t)
            h_t_minus_1 = h_t
            #break
        #output = torch.stack(output)
        #if self.batch_first:
        #    output = output.transpose(0, 1)
        return None, h_t
        #return output, h_t


    def forward_s(self, x, time, mask, time_sorted, sequence_batch_mask, hx=None):
        # x is (S, B, E) where E is input_size
        # time is (S, B)
        # Want (S, B) iteration, mask updates at t = batch[time], pre-sort time
        if self.batch_first:
            x = x.transpose(0, 1)
            time = time.transpose(0, 1)
            mask = mask.transpose(0, 1)
            sequence_batch_mask = sequence_batch_mask.transpose(1, 2) # (T, S, B)
        #seq_len, batch_size, _ = x.size()
        t_len, seq_len, batch_size = sequence_batch_mask.size()
        
        #print("sorting")
        #time_sorted = torch.sort(torch.unique(torch.ravel(time[mask])), dim=0, descending=True)[0] # Sequence First
        #print("sorted")
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_t_minus_1 = hx
        #time = torch.cat((time, time[-1:]), dim=0) # repeat first time entry
        #mask = torch.cat((mask, mask[-1:]), dim=0) # To make sure we mask both times
        h_t = hx
        h_p = hx.clone()
        output = []
        #print("Integrating")
        for i in range(t_len - 1):
            #print(i)
            t1 = time_sorted[i+1]
            t0 = time_sorted[i]
            time_mask = sequence_batch_mask[i] # (S, B)
            update_mask = torch.any((time >= t0) & mask, dim=0) # Locations where the batch sample has a time earlier than t0
            time_mask_across_sequence = torch.any(time_mask, dim=0) # (B,) Any batch sample that contains the time
            #time_mask_across_batch = torch.any(time_mask, dim=1) # (S,) Any sequence element containing a batch with a time
            #print(h_p.shape, update_mask.shape, h_t_minus_1[layer].shape)
            #ts = torch.stack([t0, t1])
            n_points = max(2, int(torch.floor((t0 - t1)/self.dt)))
            ts = torch.linspace(t0, t1, n_points, device=x.device)
            #print(ts.shape, ts)
            #h_p[update_mask] = self.integrator(self.ode_net, h_t_minus_1[update_mask], t0, t1) # Update all batches at this time
            h_p[update_mask] = self.integrator(self.ode_net, h_t_minus_1[update_mask], ts, atol=1e-3, rtol=1e-5, method='rk4')[1]
            h_p[~update_mask] = 0.0

            # Note: Should really only update positions where t > t0 

            x_t = x[time_mask] # (S_t & B_t, E)

            # Think about updating individual h_t componenets and only running rnn on those componenets
            h_t[time_mask_across_sequence] = self.rnn_cell(x_t, h_p[time_mask_across_sequence]) # Update all batches at this time at their sequence locations
            h_t[~time_mask_across_sequence] = h_p[~time_mask_across_sequence] # Otherwise update with previous integration step
            output.append(h_t)
            h_t_minus_1 = h_t
        output = torch.stack(output)
        #print("integrated", time_sorted.shape[0]-1, "steps")
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t
    

    def forward__(self, x, time, mask, hx=None):
        return
        # x is (S, B, E) where E is input_size
        # time is (S, B)
        # Want (S, B) iteration, mask updates at t = batch[time], pre-sort time
        if self.batch_first:
            x = x.transpose(0, 1)
            time = time.transpose(0, 1)
            mask = mask.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        time_sorted, time_index = torch.sort(torch.ravel(time), dim=0, descending=True) # union of all times decending
        mask_sorted = torch.ravel(mask)[time_index] # Masks at each loction
        time_index = time_index[mask_sorted] # Mask out the time index
        time_sorted = time_sorted[mask_sorted] # mask out the times
        idx2S = time_index // batch_size # Sequence number at each time point
        idx2B = time_index % batch_size # batch number at each time point
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_t_minus_1 = hx
        #time = torch.cat((time, time[-1:]), dim=0) # repeat first time entry
        #mask = torch.cat((mask, mask[-1:]), dim=0) # To make sure we mask both times
        h_t = hx
        #x = x[::-1] # reversed time-series encoding
        #time = time[::-1] # reversed time-series encoding
        output = []
        for i in range(time_sorted.shape[0]-1):
            t1 = time_sorted[i+1]
            t0 = time_sorted[i]
            x_t = x[idx2S[i+1]] # (B, E)
            mask_t = F.one_hot(idx2B[i+1], num_classes=batch_size)
    

            h_p = self.integrator(self.ode_net, h_t_minus_1, t0, t1) # Update everything across all batches
            #h_t[idx2B[i+1]] = self.rnn_cell(x_t[idx2B[i+1]], h_t[idx2B[i+1]])
            h_rnn = self.rnn_cell(x_t, h_p)
            h_t = h_rnn * mask_t[:, None] + (1 - mask_t[:, None]) * h_p

            output.append(h_t)
            h_t_minus_1 = h_t
            #print(h_t)
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
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


    
