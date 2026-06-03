import torch
from torch import nn



class StaticSpectrumBasis(nn.Module):
    """Creates sets of standard basis functions in wavelength and time and uses the latent variable for selection"""

    def __init__(self, partial_network, n_features=8, n_output=1, separable=False, sine_transform=False, **kwargs):

        super().__init__()

        self.n_features = n_features
        self.n_output = n_output
        self.separable = separable
        self.sine_transform = sine_transform
        self.neural_network = partial_network(n_features=self.n_features + 2, n_output=n_output, **kwargs)
        self.temporal_neural_network = partial_network(n_features=self.n_features + 2, n_output=n_output, **kwargs)
        self.selector_neural_network = partial_network(n_features=self.n_features + 2, n_output=n_output*2, **kwargs)
        assert self.separable
            

        self.wavelength_encoding = nn.Linear(1, self.n_features+2)
        self.latent_encoding = nn.Linear(self.n_features, self.n_features+2)
        self.time_encoding = nn.Linear(1, self.n_features+2)

        self.softmax = nn.Softmax(-1)
        


    def forward(self, time, wavelength, features):

        # Time: (* x seq_length x 1) or (* x 1)
        # Wavelength: (* x 1 x wavelength_length) or (* x 1)
        # Features: (* x n_features)
        # Typically * is batch_size x num_basis

        wavelength_shape = wavelength.shape
        time_shape = time.shape

        latent_encoding = self.latent_encoding(features[..., None, None, :]) # B, 1, 1, E -> B, 1, 1, F
        wavelength_encoding = self.wavelength_encoding(wavelength[..., None]) # B, 1, L, 1 -> B, 1, L, F
        if self.sine_transform:
            wavelength_encoding = torch.sin(wavelength_encoding) + wavelength_encoding
        time_encoding = self.time_encoding(time[..., None]) # B, S, 1, 1 -> B, S, 1, F

        selector = self.selector_neural_network(latent_encoding)
        time_selector, wave_selector = selector.reshape(*selector.shape[:-1], 2, -1).unbind(-2) # Each are B, 1, 1, n_output
        time_selector = self.softmax(time_selector)
        wave_selector = self.softmax(wave_selector)

        time_component = self.temporal_neural_network(time_encoding)
        wave_component = self.neural_network(wavelength_encoding)



        return self.neural_network(wavelength_encoding), self.temporal_neural_network(time_encoding)