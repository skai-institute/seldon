import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from TAE.util import register_module
import math

# TODO: inherit ResNet and DenseNet from MLP.

@register_module
class BatchLookup(nn.Module):
    '''wrapper to a set of modules and batch indices'''

    def __init__(self, module, *keys):

        super().__init__()
        self.module = module
        self.keys = keys

    def forward(self, batch):

        return self.module(*[batch[key] for key in self.keys])

class NormalDistriubtionQuantileMapping(nn.Module):
    '''Takes in a tensor from [-inf, inf], maps to cumulative normal, run through normal disrtibution quantile'''

    def __init__(self, device=None, dtype=None, sigma_init: float=1.0, mu_init: float=0.0):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.sigma_init = sigma_init
        self.mu_init = mu_init
        self.mu = Parameter(torch.empty(1, **factory_kwargs))
        self.sigma = Parameter(torch.empty(1, **factory_kwargs))
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.mu, self.mu_init)
        nn.init.constant_(self.sigma, self.sigma_init)

    def forward(self, x):

        x = 0.5 * (1 + torch.erf(x/2**0.5))
        x = x.clamp(min=1e-7, max=1-1e-7)
        x = self.mu + self.softplus(self.sigma) * 2**0.5 * torch.erfinv(2*x - 1)
        return x

class ExponentialDistriubtionQuantileMapping(nn.Module):
    '''Takes in a tensor from [-inf, inf], maps to sigmoid, run through exponetial disrtibution quantile'''

    def __init__(self, device=None, dtype=None, init: float = 1.0):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.lam = Parameter(torch.empty(1, **factory_kwargs))
        self.init = init
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.lam, self.init)

    def forward(self, x):

        x = self.sigmoid(x).clamp(min=1e-7, max=1-1e-7)
        x = -torch.log(1 - x) / self.softplus(self.lam)
        return x


@register_module
class ClassEmbedding(nn.Module):

    def __init__(self, num_classes, embedding_dim, scale_grad_by_freq=False):

        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_classes, embedding_dim, scale_grad_by_freq=scale_grad_by_freq)

    def forward(self, class_idx):

        return self.embedding(class_idx)



class DictModel(nn.Module):

    '''
    Essentially a module dict but the forward method return a dictionary of the results
    '''
    def __init__(self, **modules):

        super().__init__()

        self.modules = nn.ModuleDict(modules)

    def forward(self, x):

        return {key: self.modules[key](x) for key in modules}

@register_module
class NormalDistributionModel(nn.Module):

    def __init__(self, input_dim, output_dim, epsilon=1e-4):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.fc_mean = nn.Linear(input_dim, output_dim)
        self.fc_sigma = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus())

    def forward(self, x):

        mean = self.fc_mean(x)
        sigma = self.fc_sigma(x) + self.epsilon

        return mean, sigma

@register_module
class ApproximatePosterior:

    def sample(self, mean, sigma):

        eps = torch.randn_like(sigma)
        z = mean + eps * sigma

        return z
    
    def kl_divergence_(self, mean, var):

        return -0.5 * (1 + torch.log(var) - torch.square(mean) - var)
    
    def kl_divergence(self, mean, var):

        return torch.sum(self.kl_divergence_(mean, var), dim=-1)



@register_module
class VariationalComponent(nn.Module):

    def __init__(self, input_dim, output_dim, epsilon=1e-4, variational=True, scale_sigma_by_number=False):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.variational = variational
        self.epsilon = epsilon
        self.scale_sigma_by_number = scale_sigma_by_number

        self.fc_mean = nn.Linear(input_dim, output_dim)
        self.fc_sigma = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus())

    def sample(self, mean, sigma):

        eps = torch.randn_like(sigma)
        z = mean + eps * sigma

        return z

    def forward(self, x, mask=None):

        z_mean = self.fc_mean(x)
        z_sigma = self.fc_sigma(x) + self.epsilon
        if self.scale_sigma_by_number and mask is not None:
            N = mask.sum(1).reshape(-1, 1)
            z_sigma = z_sigma / torch.sqrt(N+self.epsilon)

        if self.variational:
            z_sample = self.sample(z_mean, z_sigma)
        else:
            z_sample = z_mean

        return {"z_mean": z_mean, "z_sigma": z_sigma, "z_var": torch.square(z_sigma), "z_sample": z_sample}
    
    def kl_divergence_(self, z_mean, z_var):
        if self.variational:
            return -0.5 * (1 + torch.log(z_var) - torch.square(z_mean) - z_var)
        return 0
    
    def kl_divergence(self, z_mean, z_var, weight=None):

        if self.variational:
            if weight is None:
                return -0.5 * torch.mean(1 + torch.log(z_var) - torch.square(z_mean) - z_var)
            else:
                return -0.5 * torch.mean(torch.mean(1 + torch.log(z_var) - torch.square(z_mean) - z_var, dim=1) * weight)
        else:
            return 0.0

class BinaryGELU(nn.Module):

    def __init__(self, approximate: str="none", device=None, dtype=None, init: float = 0.0):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.approximate = approximate
        self.init = init
        self.weight = Parameter(torch.empty(2, **factory_kwargs))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.constant_(self.weight, self.init)

    def forward(self, x):

        #return x + torch.sin((x - self.weight[0])*self.weight[1])
        return F.gelu(x-self.weight[0], approximate=self.approximate) - F.gelu(x-self.weight[1], approximate=self.approximate)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
    ):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError as e:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from e
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers,
        init,
        activation,
        dropout,
        norm,
        epsilon=1e-4,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.init = init
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.n_inputs = n_inputs
        self.epsilon = epsilon

        self.layers = []
        for i in range(0, self.n_layers):
            # sigma-pi, plus initialisation
            layer = nn.Linear(in_features=n_inputs, out_features=self.n_hidden)
            eval("torch.nn.init.%s(layer.weight)" % self.init)
            self.layers.append(layer)
            # then the activation function
            cmd = "nn.%s()" % self.activation
            self.layers.append(eval(cmd))
            n_inputs = self.n_hidden
            # then normalisation layers
            if not np.isclose(self.dropout, 0):
                self.layers.append(nn.Dropout(self.dropout))
            if self.norm == "batch":
                self.layers.append(nn.BatchNorm1d(self.n_hidden))
            elif self.norm == "layer":
                self.layers.append(nn.LayerNorm(self.n_hidden))
        # and a linear output at the end
        self.layers.append(nn.Linear(self.n_hidden, self.n_outputs))
        self.layers.append(eval(cmd))
        self.neuralnet = nn.Sequential(*self.layers)

        self.mean_layer = nn.Linear(self.n_outputs, self.n_outputs)
        self.std_layer = nn.Sequential(
            nn.Linear(self.n_outputs, self.n_outputs), nn.Softplus()
        )

    def forward(self, inputs):
        outputs = self.neuralnet(inputs)
        mu = self.mean_layer(outputs)
        sigma = self.std_layer(outputs) + self.epsilon
        return mu, sigma
    
class HyperMatrix(nn.Module):
    """Produce k (m x n) matricies and biases"""

    def __init__(self, input_dim, m, n, k=1, hidden_dim=None, bias=True, **resnet_kwargs):
        super().__init__()

        self.input_dim = input_dim # size of the vector that generates the matrix series
        self.m = m # size of input vector to multiply by
        self.n = n # "output dim" of the network
        self.k = k # number of matricies to produce

        self.n_out = n + (1 if self.bias else 0)

        self.output_dim = (m * self.n_out) * k

        self.bias = bias

        if hidden_dim is None:
            self.hidden_dim = self.output_dim
        else:
            self.hidden_dim = hidden_dim

        self.mlp = ResNet(
            self.input_dim, 
            self.output_dim, 
            nhid=self.hidden_dim, 
            **resnet_kwargs)

    def forward(self, x):
        # x: (B, m)
        # weights: k x m x n
        # bias (optional): k x m x 1

        weights = self.mlp(x).unsqueeze(-1).unsqueeze(-1) # (B, kxmxn_out, 1, 1)
        weights = weights.view(*x.shape[:-1], self.k, self.m, self.n_out) # (B, k, m, n_out)
        if self.bias:
            bias = weights[..., -1:] # (B, O, 1)
            weights = weights[..., :-1] # (B, O, E)
            return weights, bias
        return weights


class HyperLinear(nn.Module):
    """Learn and MLP from inputs and apply it to the inputs"""

    def __init__(self, input_dim, output_dim, hidden_dim=None, input_dim_cond=None, bias=True, **resnet_kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_cond = input_dim_cond
        self.bias = bias
        if input_dim_cond is None:
            self.input_dim_cond = input_dim

        if hidden_dim is None:
            self.hidden_dim = output_dim * (input_dim + (1 if self.bias else 0))
        else:
            self.hidden_dim = hidden_dim

        self.mlp = ResNet(
            self.input_dim_cond, 
            output_dim * (input_dim + (1 if self.bias else 0)), 
            nhid=self.hidden_dim, 
            **resnet_kwargs)

    def forward(self, x, cond=None):
        # x: (B, E)
        if cond is None:
            cond = x

        weights = self.mlp(cond).unsqueeze(-1) # (B, ExO, 1)
        weights = weights.view(*x.shape[:-1], self.output_dim, self.input_dim + (1 if self.bias else 0)) # (B, O, E)
        if self.bias:
            bias = weights[..., -1:] # (B, O, 1)
            weights = weights[..., :-1] # (B, O, E)
        x = torch.bmm(weights, x[..., None]) # (B, O, 1)
        if self.bias:
            x = x + bias # (B, O, 1)
        x = x.squeeze(-1) #(B, O)
        return x
    
@register_module
class SimpleHyperMLP(nn.Module):

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim, 
                 hyper_hidden_dim=128,
                 input_dim_cond=None, 
                 num_layers=2, 
                 activation="Softplus", 
                 bias=True):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)
        self.bias = bias
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        self.hyper_mlp = []
        for i in range(num_layers):
            self.hyper_mlp.append(
                nn.ModuleList([
                    HyperLinear(
                        input_dim=input_dim if i == 0 else hidden_dim,
                        output_dim=output_dim if i == num_layers-1 else hidden_dim,
                        input_dim_cond=input_dim_cond,
                        bias=bias,
                        hidden_dim=hyper_hidden_dim,
                        nlayers=4
                        ),
                        *[self.activation() for i in range(1) if i+1 < num_layers],
                    ])
            )
        
        self.hyper_mlp = nn.ModuleList(self.hyper_mlp)

    def forward(self, x, cond=None):

        for block in self.hyper_mlp:
            x = block[0](x, cond)
            if len(block) == 2:
                x = block[1](x)
        return x
    

class HyperMLPBlock(nn.Module):
    """Learn and MLP from inputs and apply it to the inputs"""

    def __init__(self, input_dim, output_dim, hidden_dim=None, activation=None, bias=True, norm=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim is None:
            self.hidden_dim = output_dim * (input_dim + (1 if self.bias else 0))
        else:
            self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias = bias

        if self.activation is not None:
            self.activation = getattr(nn, activation)()
        else:
            self.activation = lambda x: x
        #self.mlp_learner = nn.Linear(input_dim, input_dim * output_dim)
        self.mlp_learner = ResNet(
            input_dim, output_dim * (input_dim + (1 if self.bias else 0)), 
            nlayers=2, nhid=self.hidden_dim, 
            activation="PReLU",
            dropout=0.0, norm=norm)

    def forward(self, x, cond):
        # x: (B, E)

        weights = self.mlp_learner(x).unsqueeze(-1) # (B, ExO, 1)
        weights = weights.view(*x.shape[:-1], self.output_dim, self.input_dim + (1 if self.bias else 0)) # (B, O, E)
        if self.bias:
            bias = weights[..., -1:] # (B, O, 1)
            weights = weights[..., :-1] # (B, O, E)
        x = torch.bmm(weights, x[..., None]) # (B, O, 1)
        if self.bias:
            x = x + bias # (B, O, 1)
        x = self.activation(x.squeeze(-1)) #(B, O)
        return x
    
@register_module
class Partial:

    def __init__(self, model_class, **kwargs):

        self.model_class = model_class
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        matches = set(kwargs.keys()) & set(self.kwargs.keys())
        if len(matches):
            import sys
            print(f"WARNING: Partial wrapper for {self.model_class.__name__} detected duplicate instances of the following kwargs:\n  ", matches, "\nWill override the following:", file=sys.stderr)
            for key in matches:
                print(f"  {key}: {self.kwargs[key]} -> {kwargs[key]}", file=sys.stderr)
        model_kwargs = self.kwargs.copy()
        model_kwargs.update(kwargs)
        return self.model_class(*args, **model_kwargs)

@register_module
class MultiHeadLinearBlock(nn.Module):

    def __init__(self, in_features, out_features=None, bias=True, n_heads=1):
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        self.out_features = in_features if out_features is None else out_features
        assert in_features / n_heads == in_features//n_heads, f"n_features ({in_features}) must be divisible by n_heads ({n_heads})"
        assert self.out_features / n_heads == self.out_features//n_heads, f"n_output ({out_features}) must be divisible by n_heads ({n_heads})"

        # n_networks = n_heads
        # in_channels -> n_features
        # groups -> n_networks

        self.multihead_linear = nn.Conv1d(self.in_features, self.out_features, groups=self.n_heads, kernel_size=1, bias=bias)

    def forward(self, x):
        # Assume x is of shape (* x n_features)

        x_shape = x.shape
        x = x.reshape(x.shape[0], -1, self.in_features).transpose(-2, -1).contiguous() # (* x n_features x 1)
        return self.multihead_linear(x).transpose(-2, -1).reshape(*x_shape[:-1], self.out_features).contiguous() # (* x n_output)

        x_shape = x.shape
        x = x.reshape(-1, self.in_features, 1) # (* x n_features x 1)
        return self.multihead_linear(x).reshape(*x_shape[:-1], self.out_features) # (* x n_output)

@register_module
class FastMultiHeadLinearBlock(nn.Module):

    def __init__(self, in_features, out_features=None, bias=True, n_heads=1):
        super().__init__()

        self.in_features = in_features
        self.n_heads = n_heads
        self.out_features = in_features if out_features is None else out_features
        assert in_features / n_heads == in_features//n_heads, f"n_features ({in_features}) must be divisible by n_heads ({n_heads})"
        assert self.out_features / n_heads == self.out_features//n_heads, f"n_output ({out_features}) must be divisible by n_heads ({n_heads})"

        # n_networks = n_heads
        # in_channels -> n_features
        # groups -> n_networks

        k = int(math.sqrt(in_features))
        self.features_per_head = in_features // n_heads

        self.weight = nn.Parameter(torch.rand(1, n_heads, out_features//n_heads, self.features_per_head, requires_grad=True) * 2*k - k)
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features, requires_grad=True) * 2*k - k)
        else:
            self.bias = None

    def forward(self, x):
        # Assume x is of shape (* x n_features)

        shape = x.shape
        if self.bias is not None:
            return torch.matmul(self.weight, x.view(*shape[:-1], self.n_heads, self.features_per_head, 1)).view(*shape[:-1], self.out_features) + self.bias
        else:
            return torch.matmul(self.weight, x.view(*shape[:-1], self.n_heads, self.features_per_head, 1)).view(*shape[:-1], self.out_features)



@register_module
class MultiHeadLayerNorm(nn.Module):

    def __init__(self, n_features, n_heads=1):
        super().__init__()

        self.n_features = n_features
        self.n_heads = n_heads
        assert n_features / n_heads == n_features//n_heads, f"n_features ({n_features}) must be divisible by n_heads ({n_heads})"

        # n_networks = n_heads
        # in_channels -> n_features
        # groups -> n_networks
        self.heads = nn.ModuleList([nn.LayerNorm(self.n_features//self.n_heads) for _ in range(self.n_heads)])


    def forward(self, x):
        # Assume x is of shape (* x n_features)

        x_shape = x.shape
        x = x.reshape(x.shape[0], -1, self.n_heads, self.n_features//self.n_heads) # (* x n_features x 1)
        return torch.cat([head(x_i) for x_i, head in zip(x.unbind(-2), self.heads)], dim=-1).reshape(*x_shape)

@register_module
class SlidingBasisSuppression(nn.Module):
    """learnable vector of basis activations
    to control the number of basis functions 
    through a sliding offset on a HardTanh"""

    def __init__(self, num_basis):
        super().__init__()

        self.num_basis = num_basis

        self.controller = nn.Parameter(torch.zeros(1, 1, 1, requires_grad=True), requires_grad=True)
        slider = torch.arange(num_basis, requires_grad=False)[None, None, :]
        self.register_buffer("slider", slider)

        self.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.scaler = nn.Sigmoid()

    def forward(self):

        return self.activation((self.num_basis)*(self.scaler(self.controller) - 1) + self.slider + 1)
    
@register_module
class ResidualBlock(nn.Sequential):

    def forward(self, x):
        
        return super().forward(x) + x

@register_module
class DropoutBlock(nn.Sequential):
    # Dropouts typically come before linear layers except for the output layer: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

    def __init__(self, *layers, dropout=0.0):

        super().__init__(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return super().forward(self.dropout(x))

@register_module
class GatedLinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dim=-1, n_heads=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        Linear = nn.Linear if n_heads == 1 else Partial(MultiHeadLinearBlock, n_heads=n_heads)

        self.layer = nn.Sequential(Linear(in_features, out_features*2, bias=bias), nn.GLU(dim=dim))

    def forward(self, x):

        return self.layer(x)

@register_module
class FeedFowardLayer(nn.Module):
    # Note: Normalization Layers typically come before activation functions, typically after a linear layer https://arxiv.org/abs/1603.05027

    __constants__ = ["in_features", "out_features", "bias", "n_heads"]

    in_features: int
    out_features: int
    n_heads: int
    bias: bool


    def __init__(self, in_features, out_features, bias=True, n_heads=1, activation="ReLU", pre_activation=False, dropout=0.0, norm=None, residual=False):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.bias = bias

        activation_function = getattr(nn, activation)

        if n_heads == 1:
            linear_layer = nn.Linear(in_features, out_features, bias=bias)
        else:
            linear_layer = MultiHeadLinearBlock(in_features, out_features, bias=bias, n_heads=n_heads)
        
        if norm:
            activation_block = nn.Sequential(nn.LayerNorm({True: in_features, False: out_features}[pre_activation]), activation_function())
        else:
            activation_block = activation_function()

        layer = [linear_layer, activation_block]
        if pre_activation: layer = layer[::-1]

        self.layer = nn.Sequential(*layer)
        if dropout:
            self.layer = DropoutBlock(self.layer, dropout=dropout)

        if residual:
            if in_features != out_features:
                raise ValueError(f"For residual connections, in_features ({in_features}) must equal out_features ({out_features})")
            self.layer = ResidualBlock(self.layer)

    def forward(self, x):

        return self.layer(x)

@register_module
class MultiLayerPerceptron(nn.Module):

    def __init__(self, in_features, out_features, hidden_size, n_layers, bias=True, n_heads=1, activation="ReLU", pre_activation=False, dropout=0.0, norm=None, residual=False, skip=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.bias = bias
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        layer_constructor = Partial(FeedFowardLayer, bias=bias, n_heads=n_head, activation=activation, pre_activation=pre_activation, dropout=dropout, norm=norm)
        input_layer = layer_constructor(in_features, hidden_size, residual=False)
        hidden_layers = nn.Sequential([layer_constructor(hidden_size, hidden_size, residual=residual and not skip) for _ in range(n_layers-1)])
        if skip:
            if not residual:
                raise ValueError("MultiLayerPerceptron got skip=True but residual=False")
            hidden_layers = ResidualBlock(hidden_layers)
        if pre_activation:
            output_layer = layer_constructor(hidden_size, out_features, residual=False)
        else:
            output_layer = nn.Linear(hidden_size, out_features, bias=bias) if n_heads == 1 else MultiHeadLinearBlock(hidden_size, out_features, bias=bias, n_heads=n_heads)
        self.neural_network = nn.Sequential(input_layer, hidden_layers, output_layer)

    def forward(self, x):

        return self.neural_network(x)


@register_module
class ResidualNetwork(nn.Module):    

    def __init__(
        self,
        n_features,
        n_output,
        n_layers=2,
        n_hidden=200,
        activation="Softplus",
        dropout=0.5,
        norm=False,
        bias=True,
        n_heads=1,
        norm_first=True,
        residual_input=False,
    ):
        
        super().__init__()

        self.ninp = n_features
        self.noutp = n_output
        self.nlayers = n_layers
        self.nhid = n_hidden
        self.bias = bias
        self.n_heads = n_heads

        linear_block = nn.Linear if n_heads == 1 else Partial(MultiHeadLinearBlock, n_heads=n_heads)
        LayerNorm = nn.LayerNorm if n_heads == 1 else Partial(MultiHeadLayerNorm, n_heads=n_heads)
        self.activation_function = getattr(nn, activation)

        # Generate Input Layers
        input_layer = [linear_block(self.ninp, self.nhid, bias=bias)]
        if norm and norm_first: input_layer.append(LayerNorm(self.nhid)) # pre-norm layer
        input_layer.extend([self.activation_function(), nn.Dropout(p=dropout)]) # Activation function and dropout
        if norm and not norm_first: input_layer.append(LayerNorm(self.nhid)) # add and norm behavior, break identity relation
        input_layer = nn.Sequential(*input_layer) # create an mlp block then add a residual connection

        # Generate Hidden Layers
        hidden_layers = []
        for _ in range(self.nlayers-1):
            hidden_layer = [linear_block(self.nhid, self.nhid, bias=bias)]
            if norm and norm_first: hidden_layer.append(LayerNorm(self.nhid)) # pre-norm layer
            hidden_layer.extend([self.activation_function(), nn.Dropout(p=dropout)]) # Activation function and dropout
            hidden_layers.append(ResidualBlock(*hidden_layer)) # create an mlp block then add a residual connection
            if norm and not norm_first: hidden_layers.append(LayerNorm(self.nhid)) # add and norm behavior, break identity relation
            
        hidden_layers = nn.Sequential(*hidden_layers)

        # Generate Output Layer
        output_layer = linear_block(self.nhid, self.noutp, bias=bias)

        self.neural_network = nn.Sequential(input_layer, hidden_layers, output_layer)
        if residual_input:
            assert self.ninp == self.noutp
            self.neural_network = ResidualBlock(self.neural_network)

    def forward(self, x):

        return self.neural_network(x)



@register_module
class ResNet(nn.Module):

    def __init__(
        self,
        ninp,
        noutp,
        nlayers=2,
        nhid=200,
        activation="Softplus",
        dropout=0.5,
        init="xavier_normal_",
        norm=False,
        bias=True
    ):

        super().__init__()

        self.ninp = ninp
        self.noutp = noutp
        self.nlayers = nlayers
        self.nhid = nhid
        self.bias = bias

        self.activation_function = getattr(nn, activation)

        self.input_layer = nn.Sequential(
            nn.Linear(ninp, nhid, bias=bias), *[nn.LayerNorm(nhid) for i in range(1) if norm], self.activation_function(), nn.Dropout(p=dropout)
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(nhid, nhid, bias=bias),
                    *[nn.LayerNorm(nhid) for i in range(1) if norm],
                    self.activation_function(),
                    nn.Dropout(p=dropout),
                    
                )
                for _ in range(nlayers - 1)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(nhid, noutp, bias=bias),
        )

        #self.init_weights()

    def init_weights(self):

        for layer in [*self.input_layer, *self.hidden_layers, *self.output_layer]:
            if isinstance(layer, nn.Linear):
                if hasattr(layer, "weight"):
                    eval("torch.nn.init.%s(layer.weight)" % self.init)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

    def forward(self, x):

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x) + x
        x = self.output_layer(x)
        return x


class DenseNet(nn.Module):
    # TODO: remove un-used parameters

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers,
        init,
        activation,
        dropout=0.0,
        norm="none",
        epsilon=1e-4,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.activation = activation
        self.init = init

        dims = (n_inputs,) + (n_hidden,) * n_layers + (n_outputs,)

        sequential = [
            nn.Sequential(
                nn.Linear(dims[0], dims[1]),
                # eval("torch.nn.init.%s(layer.weight)" % self.init),
                eval("nn.%s()" % self.activation),
                nn.Dropout(p=dropout),
            )
        ]
        for t in range(len(dims) - 2):
            sequential.append(
                nn.Sequential(
                    nn.Linear(dims[t] + dims[t + 1], dims[t + 2]),
                    # eval("torch.nn.init.%s(layer.weight)" % self.init),
                    eval("nn.%s()" % self.activation),
                    nn.Dropout(p=dropout),
                )
            )

        self.sequential = nn.Sequential(*sequential)

        self.mean_layer = nn.Linear(n_outputs, n_outputs)
        self.std_layer = nn.Sequential(nn.Linear(n_outputs, n_outputs), nn.Softplus())

        for p in self.sequential.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.mean_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.std_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

    def forward(self, inputs):
        outputs = self.sequential[0](inputs)
        outputs_pre = inputs
        outputs_curr = outputs
        for t in range(len(self.sequential) - 1):
            outputs = self.sequential[t + 1](
                torch.cat((outputs_pre, outputs_curr), dim=-1)
            )
            outputs_pre = outputs_curr
            outputs_curr = outputs

        mu = self.mean_layer(outputs)
        sigma = self.std_layer(outputs) + self.epsilon

        return mu, sigma
