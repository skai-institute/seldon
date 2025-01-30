import typing
from dataclasses import dataclass
from typing import Optional

import torch
from torch import distributions, nn
from torch.distributions.dirichlet import Dirichlet

from models.decoders import BaseDecoder
from models.encoders import BaseEncoder
from models.scheduler import AnnealingLinearScheduler


class LogDirichlet(Dirichlet):
    def log_prob_log(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            (torch.log(value * (self.concentration - 1.0))).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )


@dataclass
class VAEOutput:
    likelihood: typing.Tuple[distributions.Normal, Dirichlet]
    approximate_posterior: distributions.Normal
    prior: distributions.Normal
    observation: torch.Tensor
    latent_observation: torch.Tensor
    reconstruction: torch.Tensor


class VAELabel(nn.Module):
    def __init__(
        self,
        input_shape: typing.Tuple[int, int, int],
        latent_dim: int,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reconstruction_loss: str = "mse",
        init: str = "xavier_normal_",
        epsilon: float = 1e-4,
        reconstruction_std: float = 0.1,
        scheduler_recon: typing.Optional[
            typing.Union[AnnealingLinearScheduler, typing.Callable[[], float]]
        ] = None,
        scheduler_kl: typing.Optional[
            typing.Union[AnnealingLinearScheduler, typing.Callable[[], float]]
        ] = None,
        num_classes: int = 4,
        is_encoder_label_on: bool = True,
        is_decoder_label_on: bool = True,
        is_likelihood_global_std_on: bool = True,
        use_elemental_mass: bool = True,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss

        self.epsilon = epsilon
        self.beta: typing.Optional[float] = None
        self.num_classes = num_classes
        self.use_elemental_mass = use_elemental_mass

        self.is_encoder_label_on = is_encoder_label_on
        self.is_decoder_label_on = is_decoder_label_on
        self.is_likelihood_global_std_on = is_likelihood_global_std_on

        self.scheduler_recon = scheduler_recon
        self.scheduler_kl = scheduler_kl

        self.reconstruction_std = reconstruction_std

        # schedulers and loss weights
        self.lambda_recon: typing.Optional[float] = None
        self.lambda_kl: typing.Optional[float] = None

        if hasattr(self.scheduler_recon, "constraint_bound") and (
            self.scheduler_recon.constraint_bound is None
        ):
            self.scheduler_recon.constraint_bound = self.decoder.std_to_bound(
                self.reconstruction_std
            )

        self.latent_output: nn.Module
        self.latent_output = nn.Linear(self.encoder.hidden_dims[-1], latent_dim)

        # linear projection to latent output std
        self.latent_output_std = nn.Sequential(
            nn.Linear(self.encoder.hidden_dims[-1], latent_dim), nn.Softplus()
        )

        # linear projection to reconstruction output
        self.final_output = nn.Linear(self.decoder.hidden_dims[-1], input_shape[-1])

        self.abundance_layer = nn.Softmax(dim=-1)
        self.vel_layer = nn.Softplus()

        # learnt likelihood std
        if self.is_likelihood_global_std_on:
            self.likelihood_std_raw = nn.Parameter(
                torch.ones(1, requires_grad=True) * 0.5
            )
        else:
            # TODO: this was only tested when use_elemental_mass = True
            self.likelihood_std_layer = nn.Linear(
                self.decoder.hidden_dims[-1], input_shape[-1]
            )
        self.likelihood_scale_raw = nn.Parameter(
            torch.ones(100, requires_grad=True) * 0.5
        )
        self.likelihood_scale = nn.Softplus()
        self.likelihood_std = nn.Softplus()
        for p in self.encoder.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % init)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % init)
        for p in self.latent_output.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % init)
        for p in self.final_output.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % init)
        for p in self.latent_output_std.parameters():
            if p.dim() > 1:
                eval("nn.init.%s(p)" % init)

    def observation_given_label(self, latent_observation, labels):
        self.eval()
        labels_onehot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        )
        if self.is_decoder_label_on:
            decoder_inputs = torch.cat([latent_observation, labels_onehot], axis=1)
        else:
            decoder_inputs = latent_observation

        likelihood_mean, *likelihood_dist = self.decode(decoder_inputs)
        self.train()
        return likelihood_mean, *likelihood_dist

    def encode(
        self, inputs: torch.Tensor, mask: typing.Optional[torch.Tensor] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(inputs)
        mu = self.latent_output(x)
        sigma = self.latent_output_std(x) + self.epsilon
        return mu, sigma

    def decode(
        self,
        inputs: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, distributions.Normal]:
        output_from_decoder = self.decoder(inputs)
        likelihood_mean = self.final_output(output_from_decoder)
        batch_size = inputs.shape[0]
        # likelihood_mean = self.final_output(self.decoder(inputs))
        # print(likelihood_mean.shape)
        # import pdb; pdb.set_trace()
        if not self.use_elemental_mass:
            abundance_mean = self.abundance_layer(
                likelihood_mean.reshape(batch_size, 12, -1).swapaxes(1, 2)[
                    ..., 2:
                ]  # NOTE: switch to using data.shape[0] for the first axis and -1 for the last
            )
            abundance_std = self.likelihood_scale(self.likelihood_scale_raw)
            # breakpoint()
            abundance_dist = Dirichlet(
                abundance_mean * abundance_std.reshape(1, 100, 1)
            )

            likelihood_std = self.likelihood_std(self.likelihood_std_raw) + self.epsilon
            likelihood_dis = torch.distributions.Normal(
                likelihood_mean[:, 200], likelihood_std
            )
            return likelihood_mean, likelihood_dis, abundance_dist
        else:
            if self.is_likelihood_global_std_on:
                likelihood_std = (
                    self.likelihood_std(self.likelihood_std_raw) + self.epsilon
                )
            else:
                likelihood_std = self.likelihood_std_layer(output_from_decoder)
                likelihood_std = self.likelihood_std(likelihood_std) + self.epsilon

            likelihood_dis = torch.distributions.Normal(likelihood_mean, likelihood_std)
            return likelihood_mean, likelihood_dis

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> VAEOutput:
        # mu: mean of the output of the encoder
        # sigma: sigma of the output of the encoder
        # prior: prior of the encoder
        # approximate_posterior: posterior of the encoder
        # latent_observation: samples from the encoder posterior
        labels_onehot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        )
        if self.is_encoder_label_on:
            encoder_inputs = torch.cat(
                [inputs.view([len(inputs), -1]), labels_onehot], axis=1
            )
        else:
            encoder_inputs = inputs.view([len(inputs), -1])
        mu, sigma = self.encode(encoder_inputs)

        prior, approximate_posterior, latent_observation = self.sample(mu, sigma)

        if self.is_decoder_label_on:
            decoder_inputs = torch.cat([latent_observation, labels_onehot], axis=1)
        else:
            decoder_inputs = latent_observation

        likelihood_mean, *likelihood_dis = self.decode(decoder_inputs)
        return VAEOutput(  # type: ignore
            observation=inputs,
            latent_observation=latent_observation,
            reconstruction=likelihood_mean,
            likelihood=likelihood_dis,
            approximate_posterior=approximate_posterior,
            prior=prior,
        )

    def loss_function(
        self, vae_out: VAEOutput, is_training: bool = None
    ) -> typing.Dict[str, torch.Tensor]:
        if self.reconstruction_loss == "likelihood":
            if not self.use_elemental_mass:
                structure, abundance = (
                    vae_out.observation[:, :200],
                    vae_out.observation[:, 200:],
                )
                log_pxz_s = vae_out.likelihood[0].log_prob(structure)
                log_pxz_a = vae_out.likelihood[1].log_prob(
                    abundance.reshape(-1, 10, 100).swapaxes(
                        1, 2
                    )  # NOTE: switch to using data.shape[0] for the first axis and -1 for the middle
                )
                # log_pxz = torch.concatenate((log_pxz_s, log_pxz_a), dim=-1)
                # reconstruction_loss = -torch.sum(torch.mean(log_pxz, 0))
                log_pxz_s = -torch.sum(torch.mean(log_pxz_s, 0))
                log_pxz_a = -torch.sum(torch.mean(log_pxz_a, 0))
                reconstruction_loss = log_pxz_s + log_pxz_a
            else:
                log_pxz = vae_out.likelihood[0].log_prob(vae_out.observation)
                vel_loss = -torch.sum(torch.mean(log_pxz[:, :100], 0))
                reconstruction_loss = -torch.sum(torch.mean(log_pxz, 0))
        else:
            reconstruction_loss = super().loss_function(vae_out)["loss"]

        # kullback-leibler regularizer
        log_pz = vae_out.prior.log_prob(vae_out.latent_observation)
        log_qz = vae_out.approximate_posterior.log_prob(vae_out.latent_observation)
        kl = torch.mean(torch.sum(log_qz - log_pz, -1))

        # if is_training or (self.beta is None):
        #     self.beta = self.annealing(float(reconstruction_loss))  # type: ignore
        if is_training or (self.lambda_recon is None):
            self.lambda_recon = self.scheduler_recon(float(reconstruction_loss))

        if is_training or (self.lambda_kl is None):
            self.lambda_kl = self.scheduler_kl(float(kl))

        if self.scheduler_recon.current_constraint_satisfied:
            # evaluation = reconstruction_loss + kl
            evaluation = kl
        else:
            evaluation = 1e6

        outputs = {
            "loss": self.lambda_recon * reconstruction_loss + self.lambda_kl * kl,
            "evaluation": evaluation,
            "kl_divergence": kl,
            "lambda_recon": torch.tensor(
                self.lambda_recon, dtype=torch.float32, requires_grad=False
            ),
            "lambda_kl": torch.tensor(
                self.lambda_kl, dtype=torch.float32, requires_grad=False
            ),
        }
        if not self.use_elemental_mass:
            outputs.update(
                {
                    "reconstruction_loss_s": log_pxz_s,
                    "reconstruction_loss_a": log_pxz_a,
                }
            )
        else:
            # breakpoint()
            outputs.update(
                {"reconstruction_loss": reconstruction_loss, "vel_loss": vel_loss}
            )

        return outputs

    @staticmethod
    def sample(
        mu: torch.Tensor,
        std: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> typing.Tuple[distributions.Normal, distributions.Normal, torch.Tensor]:
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        approximate_posterior = torch.distributions.Normal(mu, std)
        if n_samples is None:
            random_sample = approximate_posterior.rsample()
        else:
            random_sample = approximate_posterior.rsample((n_samples,))
        return prior, approximate_posterior, random_sample
