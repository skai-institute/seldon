import typing
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from .loss import *
import lightning.pytorch as pl
from pathlib import Path
import torch.nn.functional as F

from .encoders import GRUEncoder
from .embeddings import OneHotEmbedding
from .decoders import MLPBandDecoder
from .parametric_decoders import BazinDecoder
from TAE.util import register_module
from TAE.models.ModelNN import VariationalComponent, DictModel, ApproximatePosterior
from TAE.energy import sampler


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.fill_(0.01)

@register_module
class SELDON(nn.Module):
    """
    SELDON autoencoder for supernova light curve reconstruction.

    This model uses a GRU-ODE encoder and an MLP decoder.
    """

    def __init__(
        self,
        latent_dim=8,
        variational=True,
        encoder="GRUEncoder",
        decoder="MLP",
        uq_net=None,
        variational_epsilon: float = 1e-4,
        scale_sigma_by_number: bool = False,
        **kwargs
    ):
        
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()

        self.scale_sigma_by_number = scale_sigma_by_number
        self.variational_epsilon = variational_epsilon
        self.variational = variational
        if len(kwargs):
            import sys
            print(f"WARNING: extra keyword arguments provided but not used\n{kwargs}", file=sys.stderr)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.uq_net = uq_net
        
        self.variational_component = VariationalComponent(
            encoder.hidden_dim, 
            latent_dim, 
            variational=variational,
            epsilon=self.variational_epsilon,
            scale_sigma_by_number=self.scale_sigma_by_number)

        self.apply(init_weights)

    def forward(self, x, t, band_idx_part, band_idx_full, mask=None, time_sorted=None, sequence_batch_mask=None):
        """
        Forward pass through encoder and decoder.

        Args:
            x: (B, T1, input_dim) — input partial curve (time, flux, 1)
            t: (B, T2, 1) — full time grid for reconstruction
            band_idx_full: (B, T2, 1) — band code for each point

        Returns:
            dict with 'reconstructed', 'z_mean', 'z_logvar'
        """
        memory = self.encoder(x, band_idx_part, mask)

        latent_components = self.variational_component(memory, mask)
        z_mean = latent_components['z_mean']
        z_sample = latent_components['z_sample']
        z_var = latent_components['z_var']
        z_sigma = latent_components['z_sigma']
        if self.training:
            z = z_sample
        else:
            z = z_mean

        reconstructed, parameters = self.decoder(z, t, band_idx_full, return_parameters=True)
        #parameters = dict(zip(['basis', 'weights', 'mu', 'sigma', 'e_b', 'redshift'], parameters))
        
        out = {"reconstructed": reconstructed, "z_mean": z_mean, "z_var": z_var, 'z_sample': z_sample, "z_sigma": z_sigma, "parameters":parameters, "z":z, "memory":memory}

        if self.uq_net is not None:
            error = self.uq_net(
                torch.cat(
                    [
                        reconstructed, 
                        self.decoder.band_emb(
                            band_idx_full[:, :, 0]
                        ), 
                        z[:, None, :].expand(-1, reconstructed.shape[1], -1),
                        t,
                    ], 
                    dim=-1
                )
            )
            out['error'] = F.softplus(error)

        return out

    def encode(self, x, band_idx, mask=None):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            z = self.encoder(x, band_idx, mask)
            latent_components = self.variational_component(z, mask)
            z = latent_components['z_mean']
            return z

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)


@register_module
class SELDONEnergyClassifier(nn.Module):
    """
    SELDON autoencoder for supernova light curve reconstruction.

    This model uses a GRU-ODE encoder and an MLP decoder.
    """

    def __init__(
        self,
        latent_dim=8,
        variational=True,
        encoder="GRUEncoder",
        decoder="MLP",
        uq_net=None,
        variational_epsilon: float = 1e-4,
        scale_sigma_by_number: bool = False,
        num_classes: int = 0,
        **kwargs
    ):
        
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()

        self.scale_sigma_by_number = scale_sigma_by_number
        self.variational_epsilon = variational_epsilon
        self.variational = variational
        self.num_classes = num_classes
        self.softmax = nn.Softmax(-1)
        if len(kwargs):
            import sys
            print(f"WARNING: extra keyword arguments provided but not used\n{kwargs}", file=sys.stderr)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.uq_net = uq_net
        
        self.variational_component = VariationalComponent(
            encoder.hidden_dim, 
            latent_dim, 
            variational=variational,
            epsilon=self.variational_epsilon,
            scale_sigma_by_number=self.scale_sigma_by_number)

        self.apply(init_weights)

    def forward(self, x, t, band_idx_part, band_idx_full, mask=None, time_sorted=None, sequence_batch_mask=None, class_onehot=None, class_mask=None, return_sequence=False):
        """
        Forward pass through encoder and decoder.

        Args:
            x: (B, T1, input_dim) — input partial curve (time, flux, 1)
            t: (B, T2, 1) — full time grid for reconstruction
            band_idx_full: (B, T2, 1) — band code for each point
            class_onehot: (B, num_classes) - one-hot embedding of class labels
            class_mask: (B) - Locations for which class label should be ignored

        Returns:
            dict with 'reconstructed', 'z_mean', 'z_logvar'
        """
        if return_sequence:
            memory, memory_sequence = self.encoder(x, band_idx_part, mask, return_sequence=return_sequence)
            latent_sequence = self.variational_component(memory_sequence)
        else:
            memory = self.encoder(x, band_idx_part, mask, return_sequence=False)

        latent_components = self.variational_component(memory, mask)
        z_mean = latent_components['z_mean']
        z_sample = latent_components['z_sample']
        z_var = latent_components['z_var']
        z_sigma = latent_components['z_sigma']
        if self.training:
            z = z_sample
        else:
            z = z_mean

        if (class_onehot is None or class_mask is not None):
            class_table = torch.eye(self.num_classes, device=memory.device)
            class_matrix = class_table[None, :, :].expand(memory.shape[0], -1, -1) # batch_size x all_class_dim x true_class_dim
            energy_c_given_z = self.energy_model(torch.cat([memory[:, None, :].expand(-1, self.num_classes, -1), class_matrix], dim=-1)).sum(dim=-1) # batch_size x all_class_dim, sum gets rid of last dim
            p_c_given_z = self.softmax(energy_c_given_z)
            class_label = F.one_hot(torch.argmax(p_c_given_z, dim=-1), num_classes=self.num_classes).to(z)
            if class_onehot is None:
                z = torch.cat([z, class_label], dim=1)
            else: # Substitute the guesses from the energy model where we do not know the class 
                
                z_unknown = torch.cat([z, class_label], dim=1)
                z_true = torch.cat([z, class_onehot], dim=1)
                #import sys
                #print(z_unknown.shape, z_true.shape, class_mask.shape)

                z = torch.where(class_mask[:, None], z_unknown, z_true)
        else:
            z = torch.cat([z, class_onehot], dim=1)

        reconstructed, parameters = self.decoder(z, t, band_idx_full, return_parameters=True)
        #parameters = dict(zip(['basis', 'weights', 'mu', 'sigma', 'e_b', 'redshift'], parameters))
        
        out = {"reconstructed": reconstructed, "z_mean": z_mean, "z_var": z_var, 'z_sample': z_sample, "z_sigma": z_sigma, "parameters":parameters, "z":z, "memory":memory}

        if self.uq_net is not None:
            error = self.uq_net(
                torch.cat(
                    [
                        reconstructed, 
                        self.decoder.band_emb(
                            band_idx_full[:, :, 0]
                        ), 
                        z[:, None, :].expand(-1, reconstructed.shape[1], -1),
                        t,
                    ], 
                    dim=-1
                )
            )
            out['error'] = F.softplus(error)

        if return_sequence:
            z_s_mean = latent_sequence['z_mean']
            z_s_sample = latent_sequence['z_sample']
            z_s_var = latent_sequence['z_var']
            z_s_sigma = latent_sequence['z_sigma']
            out['z_s_mean'] = z_s_mean
            out['z_s_sample'] = z_s_sample
            out['z_s_var'] = z_s_var
            out['z_s_sigma'] = z_s_sigma

        return out

    def encode(self, x, band_idx, mask=None, class_onehot=None, class_mask=None):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            memory = self.encoder(x, band_idx, mask)
            latent_components = self.variational_component(memory, mask)
            z = latent_components['z_mean']
            if (class_onehot is None or class_mask is not None):
                class_table = torch.eye(self.num_classes, device=memory.device)
                class_matrix = class_table[None, :, :].expand(memory.shape[0], -1, -1) # batch_size x all_class_dim x true_class_dim
                energy_c_given_z = self.energy_model(torch.cat([memory[:, None, :].expand(-1, self.num_classes, -1), class_matrix], dim=-1)).sum(dim=-1) # batch_size x all_class_dim, sum gets rid of last dim
                p_c_given_z = self.softmax(energy_c_given_z)
                class_label = F.one_hot(torch.argmax(p_c_given_z, dim=-1), num_classes=self.num_classes).to(z)
                if class_onehot is None:
                    z = torch.cat([z, class_label], dim=1)
                else: # Substitute the guesses from the energy model where we do not know the class 
                    z_unknown = torch.cat([z, class_label], dim=1)
                    z_true = torch.cat([z, class_onehot], dim=1)
                    z = torch.where(class_mask, z_unknown, z_true)
            else:
                z = torch.cat([z, class_onehot], dim=1)
            return z

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)


@register_module
class SELDONEnergyClassifierDictModel(nn.Module):
    """
    SELDON autoencoder for supernova light curve reconstruction.

    This model is a bit more modular using the DictModel to incorporate many different components of the model together
    """

    def __init__(
        self,
        encoder,
        decoder,
        energy_model,
        distribution_model,
        class_embedding=None,
        from_memory=False,
        variational=True,
        uq_net=None,
        class_conditional_decoder=True,
        num_classes: int = 0,
        **kwargs
    ):
        
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()
        if len(kwargs):
            import sys
            print(f"WARNING: extra keyword arguments provided but not used\n{kwargs}", file=sys.stderr)

        self.encoder = encoder
        self.decoder = decoder
        self.distribution_model = distribution_model

        self.latent_dim = self.distribution_model.output_dim

        self.energy_model = energy_model

        self.class_embedding = class_embedding if class_embedding is not None else OneHotEmbedding(num_classes)
        self.energy_model.model.class_embedding = class_embedding
        self.uq_net = uq_net
        self.approximate_posterior = ApproximatePosterior()

        self.variational = variational
        self.from_memory = from_memory
        self.num_classes = num_classes
        self.class_conditional_decoder = class_conditional_decoder
        self.softmax = nn.Softmax(-1)

        self.apply(init_weights)

    def predict_class(self, energy_continuum, probs=False):

        batch_size, latent_dim = energy_continuum.shape
        class_idx = torch.arange(self.num_classes, device=energy_continuum.device).reshape(1, -1) # 1 x Trial
        class_trials = self.class_embedding(class_idx).expand(batch_size, -1, -1).to(energy_continuum.device) # B x Trial x Emb
        continuum_trials = energy_continuum.reshape(batch_size, 1, latent_dim).expand(-1, self.num_classes, -1) # B x Trial x Latent
        energy_inputs = torch.cat([continuum_trials, class_trials], dim=-1)
        energy_c_given_z = self.energy_model(energy_inputs).sum(dim=-1) # batch_size x Trials, sum gets rid of last dim
        if probs:
            return F.softmax(energy_c_given_z, dim=-1)
        class_idx = torch.argmax(energy_c_given_z, dim=-1)
        return class_idx

    def predict_flux_err(self, reconstructed_flux, band_idx, decoder_inputs):
        if self.uq_net is not None:
            error = F.softplus(
                    self.uq_net(
                        torch.cat(
                            [
                                reconstructed_flux, 
                                self.decoder.band_emb(
                                    band_idx[:, :, 0]
                                ), 
                                decoder_inputs[:, None, :].expand(-1, reconstructed_flux.shape[1], -1),
                                t,
                            ], 
                            dim=-1
                        )
                    )
            )
            return error
        else:
            return None




    def forward(self, x, t, band_idx_part, band_idx_full, mask=None, class_idx=None, *args, **kwargs):
        """
        Forward pass through encoder and decoder.

        Args:
            x: (B, T1, input_dim) — input partial curve (time, flux, 1)
            t: (B, T2, 1) — full time grid for reconstruction
            band_idx_full: (B, T2, 1) — band code for each point
            class_onehot: (B, num_classes) - one-hot embedding of class labels
            class_mask: (B) - Locations for which class label should be ignored

        Returns:
            dict with 'reconstructed', 'z_mean', 'z_logvar'
        """
        # In this model, x will either be a dictionary or tuple of items
        
        memory, encoded_sequence = self.encoder(x, band_idx_part, mask=mask, return_sequence=False)
        z_mean, z_sigma = self.distribution_model(memory)
        z_sample = self.approximate_posterior.sample(z_mean, z_sigma)
        
        z = z_sample if self.training and self.variational else z_mean

        if class_idx is None: # Predict the class from the energy
            class_idx = self.predict_class(memory if self.from_memory else z_sample)

        class_emb = self.class_embedding(class_idx).to(z_mean.device)
        if self.class_conditional_decoder:
            decoder_inputs = torch.cat([z, class_emb], dim=-1)
        else:
            decoder_inputs = z

        reconstructed, parameters = self.decoder(decoder_inputs, t, band_idx_full, return_parameters=True)
        elbo = self.approximate_posterior.kl_divergence(z_mean, torch.square(z_sigma))
        #parameters = dict(zip(['basis', 'weights', 'mu', 'sigma', 'e_b', 'redshift'], parameters))
        
        out = {
            "reconstructed": reconstructed, 
            "z_mean": z_mean, 
            "z_var": torch.square(z_sigma), 
            'z_sample': z_sample, 
            "z_sigma": z_sigma, 
            "parameters":parameters, 
            "z":z, 
            "memory":memory,
            "decoder_inputs": decoder_inputs,
            "class_idx": class_idx,
            "class_emb": class_emb,
            "encoding_sequence": encoded_sequence,
            "elbo": elbo,
            }

        if self.uq_net is not None:
            error = self.uq_net(
                torch.cat(
                    [
                        reconstructed, 
                        self.decoder.band_emb(
                            band_idx_full[:, :, 0]
                        ), 
                        decoder_inputs[:, None, :].expand(-1, reconstructed.shape[1], -1),
                        t,
                    ], 
                    dim=-1
                )
            )
            out['error'] = F.softplus(error)

        return out

    def encode(self, x, band_idx, mask=None, class_onehot=None, class_mask=None):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            memory, encoded_sequence = self.encoder(x, band_idx_part, mask=mask)
            z_mean, z_sigma = self.distribution_model(memory)
            z_sample = self.approximate_posterior.sample(z_mean, z_sigma)
            
            z = z_sample if self.training and self.variational else z_mean

            if class_idx is None: # Predict the class from the energy
                class_idx = self.predict_class(memory if self.from_memory else z_sample)

            class_emb = self.class_embedding(class_idx)
            if self.class_conditional_decoder:
                decoder_inputs = torch.cat([z, class_emb], dim=-1)
            else:
                decoder_inputs = z
            return decoder_inputs

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)





