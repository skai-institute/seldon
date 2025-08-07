import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


# TODO: inherit ResNet and DenseNet from MLP.

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
    ):

        super().__init__()

        self.ninp = ninp
        self.noutp = noutp
        self.nlayers = nlayers
        self.nhid = nhid

        self.activation_function = getattr(nn, activation)

        self.input_layer = nn.Sequential(
            nn.Linear(ninp, nhid), self.activation_function(), nn.Dropout(p=dropout)
        )
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(nhid, nhid),
                    self.activation_function(),
                    nn.Dropout(p=dropout),
                )
                for _ in range(nlayers - 1)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(nhid, noutp),
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
