import typing
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from .loss import *
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F

from .encoders import GRUEncoder
from .decoders import MLPBandDecoder
from .parametric_decoders import BazinDecoder
from TAE.util import register_module
from TAE.models.ModelNN import VariationalComponent




def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.fill_(0.01)


@dataclass
class AEOutput:
    observation: torch.Tensor
    reconstruction: torch.Tensor
    latent_observation: torch.Tensor
    mask: torch.Tensor


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):  # Reduce latent space to 4
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),  # Replacing ReLU with LeakyReLU
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, input_dim),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, mask):

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return AEOutput(
            observation=x,
            mask=mask,
            reconstruction=reconstructed,
            latent_observation=latent,
        )

    def loss_function(self, result):

        train_loss = self.criterion(
            result.reconstruction * result.mask,
            result.observation * result.mask,
        )

        return {"loss": train_loss}


class BandAwareAutoencoder(nn.Module):
    """
    GRU-VAE-style autoencoder for supernova light curve reconstruction.
    (Single-band version, no band_idx used anymore.)

    This model uses a GRU encoder and an MLP decoder.
    """

    def __init__(
        self,
        latent_dim=8,
        kl_weight=0.001,
        out_sample_lambda=1.0,
        smooth_weight=1.0,
        variational=True,
        encoder="GRUEncoder",
        decoder="MLP",
        embedder=None,
        loss='mse',
        is_adversarial: bool=False,
        adv_alpha: float = 0.9,
        adv_epsilon: float = 0.0005,
        variational_epsilon: float = 1e-4,
        **kwargs
    ):
        
        """
        Args:
            input_dim: number of input features (e.g., time + flux + dummy 1)
            hidden_dim: hidden size for GRU encoder
            latent_dim: dimension of latent z vector
        """
        super().__init__()

        self.is_adversarial = is_adversarial
        self.adv_alpha = adv_alpha
        self.adv_epsilon = adv_epsilon
        self.variational_epsilon = variational_epsilon
        self.variational = variational
        if len(kwargs):
            import sys
            print(f"WARNING: extra keyword arguments provided but not used\n{kwargs}", file=sys.stderr)

        if isinstance(loss, str):
            if loss == 'NLL':
                self.recon_loss = MaskedGaussianNLL()
            elif loss == 'mse':
                self.recon_loss = MaskedMSELoss()
            elif loss == "Huber":
                self.recon_loss = MaskedHuberLoss()
            else:
                raise ValueError(f"argument `loss` must be one of ('NLL', 'mse', 'Huber') but got '{self.loss}'")
        else:
            self.recon_loss = loss

        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        

        self.variational_component = VariationalComponent(
            encoder.hidden_dim, 
            latent_dim, 
            variational=variational,
            epsilon=self.variational_epsilon)

        self.kl_weight = kl_weight
        self.out_sample_lambda = out_sample_lambda
        self.smooth_weight = smooth_weight

        self.loss_method = loss

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

        latent_components = self.variational_component(memory)
        z_mean = latent_components['z_mean']
        z = latent_components['z_sample']
        z_var = latent_components['z_var']
        z_sigma = latent_components['z_sigma']

        reconstructed = self.decoder(z, t, band_idx_full)

        return {"reconstructed": reconstructed, "z_mean": z_mean, "z_var": z_var, 'z_sample': z, "z_sigma": z_sigma}

    def embed(self, batch):

        with torch.no_grad():
            embedding = self.embedder(batch)
            return embedding

    def encode(self, x, band_idx, mask=None):
        """
        Encode input sequence into latent z.
        """
        with torch.no_grad():
            z = self.encoder(x, band_idx, mask)
            latent_components = self.variational_component(z)
            z = latent_components['z_mean']
            return z

    def decode(self, z, t, band_idx):
        """
        Decode latent z into flux values along time grid.
        """
        with torch.no_grad():
            return self.decoder(z, t, band_idx)

    def loss_function(self, out, batch):
        """
        Compute total loss = in-sample MSE + (out-of-sample MSE) + KL divergence + smoothness penalty.
        """

        z_mean=out["z_mean"]
        z_var=out["z_var"]

        reconstruction_loss = self.recon_loss(out, batch)
        kl = self.variational_component.kl_divergence(z_mean, z_var)

        total_loss = (
            torch.mean(reconstruction_loss) + self.kl_weight * kl
        )

        return {"loss": total_loss, 'squared_error': reconstruction_loss, 
                'kl_divergence': kl, "reconstruction_loss":torch.mean(reconstruction_loss)}

    def training_step(self, batch, batch_idx):

        if self.is_adversarial:
            delta_t = torch.tensor(0.0, requires_grad=True) # time-translation invariance
            delta_log_A = torch.tensor(0.0, requires_grad=True) # scale invariance

            #batch["x_input"][:, :, 0] = batch["x_input"][:, :, 0] + delta_t
            batch["x_input_full"] = torch.cat(
                (
                    batch["x_input_full"][..., :1] + delta_t, 
                    batch["x_input_full"][..., 1:2]*torch.exp(delta_log_A), 
                    batch["x_input_full"][..., 2:]
                ), 
                dim=-1)
            batch["time_full"] = batch["time_full"] + delta_t
            batch['flux_full'] = batch["flux_full"] * torch.exp(delta_log_A)
            batch['flux_err_full'] = batch["flux_err_full"] * torch.exp(delta_log_A)

            out = self(
                x=batch["x_input_full"],
                t=batch["time_full"],
                band_idx_part=batch["band_idx_full"],
                band_idx_full=batch["band_idx_full"],
                mask=batch["pad_mask_full"],
                time_sorted=batch['time_sorted'],
                sequence_batch_mask=batch['sequence_batch_mask'],
            )

            loss = self.loss_function(out, batch)

        else:
            out = self(
                x=batch["x_input"],
                t=batch["time_full"],
                band_idx_part=batch["band_idx"],
                band_idx_full=batch["band_idx_full"],
                mask=batch["in_sample_mask"],
                time_sorted=batch['time_sorted'],
                sequence_batch_mask=batch['sequence_batch_mask'],
            )

            loss = self.loss_function(out, batch)


        if self.is_adversarial:
            loss["loss"].backward(retain_graph=True)
            batch_adv = {}
            batch_adv.update(batch)
            
            batch_adv['flux_full'] = out['reconstructed'].clone()

            dt = torch.clamp(torch.sign(delta_t.grad.data)*torch.exp(torch.randn(1)[0])*self.adv_epsilon, max=0.5, min=-0.5)
            dA = torch.exp(torch.clamp(torch.sign(delta_log_A.grad.data)*torch.exp(torch.randn(1)[0])*self.adv_epsilon, max=0.5, min=-0.5))

            batch_adv["x_input"] = torch.cat(
                (
                    batch_adv["x_input"][..., :1]+dt, 
                    batch_adv["x_input"][..., 1:2]*dA, 
                    batch_adv["x_input"][..., 2:]
                ), 
                dim=-1)
            batch_adv["time_full"] = batch_adv["time_full"].clone() + dt
            batch_adv["flux_full"] = batch_adv["flux_full"].clone() * dA
            batch_adv["flux_err_full"] = batch_adv["flux_err_full"].clone() * dA


            out_adv = self.forward(
                x=batch_adv["x_input"], 
                t=batch_adv["time_full"], 
                band_idx_part=batch_adv["band_idx"], 
                band_idx_full=batch_adv["band_idx_full"], 
                mask=batch_adv["in_sample_mask"],
                time_sorted=batch_adv['time_sorted'],
                sequence_batch_mask=batch_adv['sequence_batch_mask'],)
            
            loss_adv = self.loss_function(out_adv, batch_adv)

            loss['loss'] = self.adv_alpha * loss['loss'] + (1 - self.adv_alpha) * loss_adv['loss']

        return loss



@register_module             
class StudentGRUMasked(BandAwareAutoencoder):
    """
    Student model (plain nn.Module) that re-uses BandAwareAutoencoder for the
    student network and adds a latent-space KD term against a *frozen* teacher.
    """

    def __init__(self, teacher_ckpt: str = None, lambda_kd: float = 0.7, **ae_cfg):
        # -------------------------- student nets -----------------------------
        super().__init__(**ae_cfg)
        self.lambda_kd = lambda_kd

        # -------------------------- load teacher -----------------------------
        if teacher_ckpt and Path(teacher_ckpt).exists():
            ckpt        = torch.load(teacher_ckpt, map_location="cpu")
            raw_dict    = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            # strip the Lightning prefix  ("model.") so keys match exactly
            clean_dict  = {k.replace("model.", ""): v for k, v in raw_dict.items()}

            teacher     = BandAwareAutoencoder(**ae_cfg)
            missing, unexpected = teacher.load_state_dict(clean_dict, strict=False)
            print(f"✓ teacher loaded  (missing {len(missing)}, unexpected {len(unexpected)})")

            teacher.eval().requires_grad_(False)
            self.teacher = teacher
        else:
            print("Teacher checkpoint not found - KD disabled")
            self.teacher   = None
            self.lambda_kd = 0.0            # disable KD gracefully

    # -------------------------------------------------------------------------
    def training_step(self, batch, _batch_idx=0):
        band_part = batch.get("band_idx_part", batch["band_idx"])

        # ---------------- student forward ------------------------------------
        out_s = self(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=band_part,
            band_idx_full=batch["band_idx_full"],
            mask=batch["in_sample_mask"],
        )
        losses = self.loss_function(out_s, batch)        # contains "loss"
        loss   = losses["loss"]

        # ---------------- KD term --------------------------------------------
        if self.teacher is not None:
            with torch.no_grad():
                out_t = self.teacher(
                    x=batch["x_input"],
                    t=batch["time_full"],
                    band_idx_part=band_part,
                    band_idx_full=batch["band_idx_full"],
                    mask=batch["in_sample_mask"],
                )
            kd = F.mse_loss(out_s["z_mean"], out_t["z_mean"])
            loss += self.lambda_kd * kd
            losses["kd_loss"] = kd

        losses["loss"] = loss
        return losses
