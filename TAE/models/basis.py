'''
basis.py

Collection of alternative basis functions
for parametric basis decoders
'''

import torch
from torch import nn
import torch.nn.functional as F


class DoubleSigmoid(nn.Module):

    DOF = 5

    def __init__(self):

        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, t, amplitude, rise_rate, decline_rate, offset, width):

        rise = self.sigmoid((t - (offset - width/2))*rise_rate)
        fall = self.sigmoid(decline_rate*(t - (offset + width/2)))
        return amplitude * rise * (1 - fall)
    
class DoubleSigmoidBasis(nn.Module):

    def __init__(self):
        super().__init__()
        self.basis = DoubleSigmoid()

    def forward(self, t, parameters):

        weight, r1, r2, mu, width = parameters

        basis = self.basis(t, weight, r1, r2, mu, width)
        f = torch.mean(basis, dim=-1, keepdim=True)
        return f

class FourierActivation(nn.Module):

    def forward(self, x):

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TimeBasis(nn.Module):

    def __init__(self, partial_network, n_features, n_output=1, embedding_expansion=1, **kwargs):

        super().__init__()

        self.basis = DoubleSigmoid()
        self.n_features = n_features
        self.n_output = n_output
        self.dof = self.basis.DOF
        self.embedding_expansion = embedding_expansion
        embedding_dim = (self.n_features + 2) * self.embedding_expansion

        self.neural_network = nn.ModuleList([partial_network(n_features=embedding_dim, n_output=n_output, **kwargs) for _ in range(self.dof)]) # Note: can make this faster by switching to n_output=dof*n_output and reshape to (..., n_output, dof) then unbind, does not work since n_output=n_heads
        self.latent_encoding = nn.Linear(self.n_features, embedding_dim, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()


    def forward(self, time, features):

        t = time[..., None] # B x S x 1 x 1
        latent_enc = self.latent_encoding(features[..., None, None, :]) # B x 1 x 1 x n_features -> emb_dim

        amplitude_, rise_rate_, decline_rate_, offset_, width_ = [nn(latent_enc.clone()) for nn in self.neural_network] # 5 (B x 1 x 1 x n_output)

        amplitude = self.sigmoid(amplitude_)
        rise_rate = self.softplus(rise_rate_)
        decline_rate = self.softplus(decline_rate_)
        offset = offset_ - torch.mean(offset_, dim=(-1,), keepdims=True)
        width = self.softplus(width_)

        basis = self.basis(t, amplitude, rise_rate, decline_rate, offset, width)

        return basis



class WavelengthSpectrumBasis(nn.Module):
    '''Like the spectrum basis but only the wavelength dependent part'''
    def __init__(self, partial_network, n_features=8, n_output=1, embed_freqs=False, embedding_expansion=1, **kwargs):

        super().__init__()

        self.n_features = n_features
        self.n_output = n_output
        self.embedding_expansion = embedding_expansion
        self.embed_freqs = embed_freqs
        embedding_dim = (self.n_features + 2) * self.embedding_expansion
        self.neural_network = partial_network(n_features=embedding_dim, n_output=n_output, **kwargs)

        self.wavelength_encoding = nn.Sequential(nn.Linear(1, embedding_dim), FourierActivation(), nn.Linear(embedding_dim*2, embedding_dim))

        self.latent_encoding = nn.Linear(self.n_features, embedding_dim, bias=False)


    def forward(self, wavelength, features):

        # wavelength: N x * x 1
        # features: N x n_features
        if self.embed_freqs:
            wave = 1/wavelength[..., None]
        else:
            wave = wavelength[..., None]
        latent = features[..., None, None, :]

        return self.neural_network(self.wavelength_encoding(wave)+self.latent_encoding(latent))



class SpectrumBasis(nn.Module):

    def __init__(self, partial_network, n_features=8, n_output=1, separable=False, sine_transform=False, time_impulse_first=False, embed_freqs=False, embedding_expansion=1, use_modular_basis=False, **kwargs):

        super().__init__()

        self.n_features = n_features
        self.n_output = n_output
        self.separable = separable
        self.sine_transform = sine_transform
        self.time_impulse_first = time_impulse_first
        self.embedding_expansion = embedding_expansion
        self.embed_freqs = embed_freqs
        self.use_modular_basis = use_modular_basis

        if self.use_modular_basis:
            self.wave_basis = WavelengthSpectrumBasis(partial_network, n_features=n_features, n_output=n_output, embed_freqs=embed_freqs, embedding_expansion=embedding_expansion, **kwargs)
            self.time_basis = TimeBasis(partial_network, n_features=n_features, n_output=n_output, embedding_expansion=embedding_expansion, **kwargs)

        else:
            embedding_dim = (self.n_features + 2) * self.embedding_expansion
            self.neural_network = partial_network(n_features=embedding_dim, n_output=n_output, **kwargs)
            if self.separable:
                self.temporal_neural_network = partial_network(n_features=embedding_dim, n_output=n_output, **kwargs)

            self.wavelength_encoding = nn.Linear(1, embedding_dim)
            if self.sine_transform:
                self.wavelength_encoding2 = nn.Linear(1, embedding_dim) # NOTE: SOME MODELS NEED THIS SET!
                #self.wavelength_encoding2 = nn.Identity()
            self.latent_encoding = nn.Linear(self.n_features, embedding_dim, bias=False)
            self.time_encoding = nn.Linear(1, embedding_dim, bias=False)
        


    def forward(self, time, wavelength, features):

        # Time: (* x seq_length x 1) or (* x 1)
        # Wavelength: (* x 1 x wavelength_length) or (* x 1)
        # Features: (* x n_features)
        # Typically * is batch_size x num_basis

        if self.use_modular_basis:
            return self.wave_basis(wavelength, features.clone()), self.time_basis(time, features.clone())

        wavelength_shape = wavelength.shape
        time_shape = time.shape
        if not self.separable:
            w_enc = self.wavelength_encoding(1/wavelength[..., None] if self.embed_freqs else wavelength[..., None])
            time_enc = self.time_encoding(time[..., None])
            if self.time_impulse_first:
                time_enc = F.sigmoid(time_enc) + (1 - F.sigmoid(time_enc))
            if self.sine_transform:
                w_enc = torch.sin(w_enc) + self.wavelength_encoding2(wavelength[..., None])
            input_encoding = w_enc + time_enc + self.latent_encoding(features[..., None, None, :])
            return self.neural_network(input_encoding), None
        else:
            w_enc = self.wavelength_encoding(1/wavelength[..., None] if self.embed_freqs else wavelength[..., None])
            time_enc = self.time_encoding(time[..., None])
            if self.time_impulse_first:
                time_enc = F.sigmoid(time_enc) + (1 - F.sigmoid(time_enc))
            if self.sine_transform:
                # Removed ading back wavelength scale
                w_enc = torch.sin(w_enc) + self.wavelength_encoding2(wavelength[..., None])
            wavelength_encoding = w_enc + self.latent_encoding(features[..., None, None, :])
            time_encoding = time_enc + self.latent_encoding(features[..., None, None, :])
            return self.neural_network(wavelength_encoding), self.temporal_neural_network(time_encoding)


class FullSpectrumBasis(nn.Module):
    '''Like the spectrum basis but the partial network acts on the entire wavelength vector
    rather than on specific points'''

    def __init__(self, partial_network, latent_dim=8, wavelength_dim=100, n_output=1, **kwargs):

        super().__init__()

        self.n_features = wavelength_dim * n_output
        self.wavelength_dim = wavelength_dim
        self.n_basis = n_output
        self.n_output = n_output * wavelength_dim
        self.neural_network = partial_network(n_features=self.n_features, n_output=self.n_output, **kwargs)
        self.wavelength_encoding = nn.Linear(wavelength_dim, self.n_features)
        self.latent_encoding = nn.Linear(latent_dim, self.n_features, bias=False)
        self.time_encoding = nn.Linear(1, self.n_features, bias=False)


    def forward(self, time, wavelength, features):

        # Time: (* x seq_length x 1)
        # Wavelength: (* x 1 x wavelength_length)
        # Features: (* x n_features)
        # Typically * is batch_size 

        input_encoding = self.wavelength_encoding(wavelength) + self.latent_encoding(features[..., None, :]) + self.time_encoding(time)
        return self.neural_network(input_encoding).reshape(*input_encoding.shape[:-1], self.wavelength_dim, self.n_basis), None # (* x seq_len x wavelength_dim x n_basis)

