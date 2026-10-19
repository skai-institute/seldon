import torch
from torch import nn
import lightning.pytorch as pl
from TAE.experiments.experiment import Experiment
from TAE.models.loss import *
import typing
import torch.nn.functional as F
from TAE.models.factory import get_module

from TAE.models.metrics_results import MetricsEvaluator, Metrics
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.visualizations.latent_space import plot_latents
import numpy as np
import matplotlib.pyplot as plt
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
from TAE.visualizations.percentile_forecasting import percentile_forecast
from TAE.energy import sampler
from TAE.datasets.utils import BAND_IDX_MAP



class EnergyModel(nn.Module):

    def __init__(self, n_features, n_hidden):

        super().__init__()
        self.model = nn.Sequential(nn.Linear(n_features, n_hidden), nn.GELU(), nn.Linear(n_hidden, 1))

    def forward(self, x):

        return self.model(x)

        
@register_module
class SELDONEnergyExperiment(Experiment):
    """
    LightningModule wrapper for training VAE-based SN models.
    Computes reconstruction loss, KL, smoothness. Logs visuals and metrics.
    """

    def __init__(self, 
                 model: pl.LightningModule = None,
                 params: typing.Dict[str, typing.Any] = None,
                 data=None,
                 metrics=None,
                 kl_weight: float = 0.001,
                 loss = "FlexibleGaussianLoss",
                 group_latent_by_class: bool = False,
                 weight_bands=True,
                 weight_classes=True,
                 inverse_transform=True,
                 scale_delta_by=None,
                 tune_parameters=True,
                 energy_weight=1.0,
                 inverse_scale_reconstruction=True,
                 true_redshift_rate=0.0,
                 fit_redshift=False,
                 successive_z_loss = False,
                 *args, **kwargs):

        super().__init__(model, params, data, metrics)
        self.configure_loss(loss)
        self.kl_weight = kl_weight
        self.fit_redshift = fit_redshift
        self.group_latent_by_class = group_latent_by_class
        self.weight_bands = weight_bands
        self.weight_classes = weight_classes
        self.inverse_transform = inverse_transform
        self.tune_parameters = tune_parameters
        self.scale_delta_by = scale_delta_by
        self.energy_weight = energy_weight
        self.inverse_scale_reconstruction = inverse_scale_reconstruction
        energy_dim = self.model.decoder.band_emb_dim*(self.model.decoder.num_basis*(self.model.decoder.dof-3))+3
        self.energy_model = EnergyModel(self.model.latent_dim+energy_dim, 4*(self.model.latent_dim+energy_dim))
        self.sampler = sampler.Sampler(self.energy_model, latent_dim=self.model.latent_dim, condition_dim=energy_dim, sample_size=self.data.batch_size)
        self.model.energy_model = self.energy_model
        self.true_redshift_rate = true_redshift_rate
        self.best_val = 1e32
        self.successive_z_loss = successive_z_loss


        assert scale_delta_by is None or isinstance(scale_delta_by, str), f"scale_delta_by must be str or None but got {scale_delta_by}"


    def configure_loss(self, loss_):

        if isinstance(loss_, str):
            if loss_ == "Huber":
                self.recon_loss = FlexibleHuberLoss()
            elif loss_ == "FlexibleGaussianLoss":
                self.recon_loss = FlexibleGaussianLoss()
            else:
                raise ValueError(f"argument `loss` must be one of ('Huber') but got '{loss_}'")
        else:
            self.recon_loss = loss_

    def configure_optimizers(self):
        """optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU."""
        # self.params["optimizer"]["config"]["model"] = self.model
        optimizer_class = get_module(
            self.params["optimizer"]["name"],
            self.params["optimizer"]["config"],
        )
        optimizer = optimizer_class(self.model)
        return optimizer

    def get_unknown_class_mask(self, class_names, device=None):

        if device is None:
            device = self.device
        mask = torch.tensor(np.array([name == "Unknown" for name in class_names]), device=device)

        return mask

    def forward(self, batch):
        """
        Forward pass: reconstruct flux values over full time grid.
        """
       
        return self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"],
            mask=batch["pad_mask_part"],
            return_sequence=self.successive_z_loss
        )
    
    def __group_latent_by_class(self, out, batch):

        z_mean=out["z_mean"]
        class_names = batch['class_name']
        class_set = set(class_names)
        class_map = dict(zip(class_set, range(len(class_set))))
        class_idx = torch.tensor(list(map(lambda x: class_map[x], class_names)), dtype=torch.long)
        class_onehots = F.one_hot(class_idx, num_classes=len(class_set)).to(z_mean.device) # (B, N_classes)
        class_N = class_onehots.sum(dim=0, keepdim=True) # number of class examples in each minibatch

        centers = z_mean[:, None, :] * class_onehots[:, :, None] # (B, N_classes, N_latent)
        centers = centers.sum(dim=0, keepdim=True) # (1, N_classes, N_latent)
        centers = centers / class_N[:, :, None] # (1, N_classes, N_latent)
        z_mean = z_mean[:, None, :] * class_onehots[:, :, None] - centers * class_onehots[:, :, None] # (B, N_classes, N_latent)
        z_mean = z_mean.sum(1) # (B, N_latent), sum over dim 1 just extracts the one_hot values
        return z_mean
    
    def parameter_loss(self, parameters):
        if 'mu' in parameters:
            mu = parameters['mu'] # (B, T, N, K) or (B, T, K)
        else:
            mu = parameters['offset']

        if 'weights' in parameters:
            weights = parameters['weights']
        else:
            weights = parameters['amplitude']

        mu_mean = mu.mean(dim=-1, keepdim=True) # center in basis-space
        mu_diff = (mu - mu_mean)
        #mu_loss = (mu_diff**2).mean(3).mean(2)[:, :, None]
        weight_loss = torch.abs(weights)
        param_loss = torch.mean(weight_loss * mu_diff**2, dim=(-1, -2))[:, :, None]
        #import sys
        #print(mu.shape, pad_mask.shape, file=sys.stderr)
        #mu_loss = (mu**2).mean((2,3))[:, :, None]
        param_loss = param_loss * 1e-2

        return param_loss

    def energy_loss(self, out, batch):

        real_z = out['z_mean'].clone()
        b = real_z.shape[0]
        basis_parameters = {key: value[:, 0, :].reshape(b, -1) for key, value in out['parameters'].items()}
        basis_parameters = torch.cat([value for value in basis_parameters.values()], dim=1)
        #real_z = torch.cat([real_z, basis_parameters], dim=1)
        
        small_noise = torch.randn_like(real_z) * 0.005
        real_z.add_(small_noise).clamp_(min=-5.0, max=5.0)
        fake_z = self.sampler.sample_new_exmps(steps=60, step_size=10, device=real_z.device, condition=basis_parameters)

        real_z = torch.cat([real_z, basis_parameters], dim=-1)
        fake_z = torch.cat([fake_z, basis_parameters], dim=-1)
        inp_latent = torch.cat([real_z, fake_z], dim=0)
        real_out, fake_out = self.energy_model(inp_latent).chunk(2, dim=0)

        # Calculate losses
        reg_loss = 0.1 * (real_out ** 2 + fake_out ** 2)#.mean()
        cdiv_loss = fake_out - real_out#.mean()
        energy_loss = reg_loss + cdiv_loss

        return energy_loss
    
    def loss_function(self, out, batch):
        """
        Compute total loss = in-sample MSE + (out-of-sample MSE) + KL divergence + smoothness penalty.
        """

        if self.true_redshift_rate > 0 and self.training: # If available, use the true redshifts instead of the inferred ones with some replacement rate
            valid_mask = (batch['redshift'] > 0)
            random_mask = torch.rand((len(out['reconstructed']),), device=out['reconstructed'].device) < self.true_redshift_rate
            total_mask = random_mask & valid_mask
            reconstruction_true_redshift, parameters_true_redshift = self.model.decoder(out['z'], batch['time_full'], batch['band_idx_full'], return_parameters=True, redshift=batch['redshift'])
            reconstruction = torch.where(total_mask[:, None, None], reconstruction_true_redshift, out['reconstructed'])
            parameters = {}
            for key in out['parameters']:
                parameters[key] = torch.where(total_mask, parameters_true_redshift[key], out['parameters'][key])
            out.update({'reconstructed':reconstruction, 'parameters':parameters})

            if self.model.uq_net is not None:
                error = self.model.uq_net(
                    torch.cat(
                        [
                            reconstruction, 
                            self.model.decoder.band_emb(
                                batch['band_idx_full'][:, :, 0]
                            ), 
                            out['z'][:, None, :].expand(-1, reconstruction.shape[1], -1),
                            batch['time_full'],
                        ], 
                        dim=-1
                    )
                )
                out.update({'error': F.softplus(error)})

        # Energy Based Loss Component
        energy_loss = self.energy_loss(out, batch)

        # Old SELDON Loss
        delta_scaler = None
        if self.scale_delta_by is not None:
            delta_scaler = batch[self.scale_delta_by]

        
        input, target, sigma = out["reconstructed"], batch["flux_full"], batch["flux_err_full"]
        if not self.inverse_scale_reconstruction: # This will make log-transformed decoded fluxes predict the original flux in the true flux space
            scale = self.data.dataset.flux_normalizer.max_value
            input = (input + 0.5) / scale - 0.5 # we want predicted scaled flux f is flux_norm(semilog(F))
            sigma = sigma / scale
        if self.inverse_transform:
            sigma_pred = sigma
            if 'error' in out:
                sigma_pred = out['error']
            input, sigma_pred = self.data.dataset.inverse_transform(input, sigma_pred)
            #input = input + torch.randn_like(sigma_pred) * sigma_pred
            target, sigma_true = self.data.dataset.inverse_transform(target, sigma)
            if 'error' in out:
                sigma = torch.sqrt(sigma_true**2 + sigma_pred**2)
            else:
                sigma = sigma_true


        reconstruction_loss = self.recon_loss(input, target, sigma, delta_scaler) # (B, S, E)
        
        if self.weight_bands:
            reconstruction_loss *= batch["band_weight"][:, :, None]

        z_mean, z_var = out["z_mean"], out["z_var"]
        if self.group_latent_by_class:
            z_mean = self.__group_latent_by_class(out, batch)
        
        if self.successive_z_loss:
            # Would like to map sample_z_{i-1} -> sample_z_{i} but avoid confusion in offset parameter by removing first z component
            # True z is z_s[-1]
            partial_mask = batch["pad_mask_part"][:, 1:] * batch["pad_mask_part"][:, :-1]
            z_s_mean = out['z_s_mean'][:, 1:, 1:].detach() # Post-samples, detach as to not push the current z into the past
            z_s_sample = out['z_s_sample'][:, :-1, 1:] # Pre-samples
            z_s_var = out['z_s_var'][:, :-1, 1:] + out['z_s_var'][:, 1:, 1:].detach()
            successive_z_loss = (0.5*(z_s_mean - z_s_sample)**2/(z_s_var + 1e-4) + 0.5*torch.log(2*np.pi*(z_s_var+1e-4)))
            successive_z_loss = (successive_z_loss * partial_mask[:, :, None]).sum(1)/partial_mask[:, :, None].sum(1)
        else:
            successive_z_loss = torch.zeros_like(batch["class_weight"][:, None])
        kl_loss = self.model.variational_component.kl_divergence_(z_mean, z_var) # (B, Z)

        if self.weight_classes:
            reconstruction_loss *= batch["class_weight"][:, None, None]
            kl_loss *= batch["class_weight"][:, None]
            energy_loss *= batch["class_weight"][:, None]
            successive_z_loss *= batch["class_weight"][:, None]

        # Mean over Padding Mask
        padding_mask = batch["pad_mask_full"].unsqueeze(-1)
        reconstruction_loss = (padding_mask * reconstruction_loss).sum(1) / (padding_mask.sum(1))

        total_loss = (
            reconstruction_loss.mean() + self.kl_weight * kl_loss.mean() + self.kl_weight*successive_z_loss.mean()
        )

        if self.tune_parameters:
            param_loss = self.parameter_loss(out['parameters'])
            total_loss += param_loss.mean()
        total_loss += energy_loss.mean() * self.energy_weight

        if self.fit_redshift:
            valid_mask = (batch['redshift'] > 0)
            redshift_true = batch['redshift'][valid_mask]
            redshift_pred = out['parameters']['redshift'][valid_mask]
            if self.weight_classes:
                class_weight = batch["class_weight"][valid_mask]
            else:
                class_weight = 1.0
            redshift_loss = ((redshift_true - redshift_pred)**2 * class_weight).mean()
            total_loss += redshift_loss
        else:
            redshift_loss = 0.0


        return {"loss": total_loss, 'squared_error': reconstruction_loss, "successive_z_loss": successive_z_loss.mean(),
                'kl_divergence': kl_loss.mean(), "reconstruction_loss":reconstruction_loss.mean(), 'energy_loss':energy_loss.mean(), 'redshift_loss':redshift_loss}

    def plot_band_embeddings(self):

        if not hasattr(self.model.decoder, 'lambda_grid'):
            return None

        #IDX2NAME = {0: "g", 1: "r", 2: "i", 3: "z", 4: "y", 5: "u"}
        BAND_COL = ["#2da328ff", "#e41a1c", "#ff7f00", "#333333", "#984ea3", "#377eb8"] + [None]*len(BAND_IDX_MAP)
        IDX2NAME = dict(zip(BAND_IDX_MAP.values(), BAND_IDX_MAP.keys()))

        band_idx = torch.arange(len(BAND_IDX_MAP), device=self.device)[:, None, None]
        band_emb = self.model.decoder.get_band_embedding(band_idx).cpu().detach().numpy()
        wave = self.model.decoder.lambda_grid[0, 0, :].cpu().detach().numpy()

        fig = plt.figure()
        for i in range(len(BAND_IDX_MAP)):
            plt.step(wave, band_emb[i, 0, :, 0]/band_emb[i, 0, :, 0].max(), label=IDX2NAME[i], color=BAND_COL[i])

        plt.legend()
        plt.xlabel('Wavelength Grid')
        plt.ylabel('Normalized Band Embedding')

        return fig

    def training_step(self, batch, batch_idx):
        #torch.compiler.cudagraph_mark_step_begin()
        out = self(batch)

        results= self.loss_function(out, batch)
        self.log("train_loss", results["loss"], sync_dist=True)
        self.log("train_recon", results['reconstruction_loss'], sync_dist=True)
        self.log("train_kl", results['kl_divergence'], sync_dist=True)
        self.log("train_energy", results['energy_loss'], sync_dist=True)
        self.log("train_redshift", results['redshift_loss'], sync_dist=True)
        self.log("train_successive_z", results['successive_z_loss'], sync_dist=True)
        return results

    def validation_step(self, batch, batch_idx):
        #torch.compiler.cudagraph_mark_step_begin()
        out = self(batch)

        loss = self.loss_function(out, batch)
        self.log("val_loss", loss['loss'], sync_dist=True)
        self.log("val_recon", loss['reconstruction_loss'], sync_dist=True)
        self.log("val_kl", loss['kl_divergence'], sync_dist=True)
        self.log("val_energy", loss['energy_loss'], sync_dist=True)
        self.log("val_redshift", loss['redshift_loss'], sync_dist=True)
        self.log("val_successive_z", loss['successive_z_loss'], sync_dist=True)


        result = {"loss": loss["loss"], "out": out, "batch": batch,
                'chi2': loss['squared_error'], "reconstruction_loss":loss["reconstruction_loss"],
                "kl_divergence":loss['kl_divergence']}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        try:
            self.log("val_epoch_sentinel", torch.tensor(1.0, device=self.device), on_epoch=True, sync_dist=True)

            outputs = self.validation_step_outputs
            if len(outputs) == 0:
                return

            result = outputs[0]  # just visualize one batch
            loss_val = result["loss"]
            out = result["out"]
            batch = result["batch"]

            val_loss = torch.stack([o['loss'] for o in outputs]).mean()
            val_recon = torch.stack([o['reconstruction_loss'] for o in outputs]).mean()
            val_kl = torch.stack([o['kl_divergence'] for o in outputs]).mean()
            self.log("val_loss", val_loss, sync_dist=True)
            self.log("val_recon", val_recon, sync_dist=True)
            self.log("val_kl", val_kl, sync_dist=True)

            for i in range(self.model.latent_dim):
                self.logger.experiment.add_histogram(f"z_{i}", out['z_mean'][:, i], self.current_epoch)

            for parameter in out['parameters']:
                self.logger.experiment.add_histogram(parameter, out['parameters'][parameter].view(-1), self.current_epoch)

            #return
            if self.metrics is not None:
                metrics = self.metrics(outputs[0])
                for metric in metrics:
                    #self.log(metric, metrics[metric])
                    values = metrics[metric]
                    if 'Max' in metric or "FE" in metric:
                        values = torch.log10(values)
                    if torch.any(torch.isfinite(values)):
                        values = values[torch.isfinite(values)]
                        self.logger.experiment.add_histogram(metric, values, self.current_epoch)

            if self.current_epoch == 30:
                    self.best_val = 1e32

            if val_loss.item() < self.best_val:
                if self.current_epoch < 0:
                    return
                else:
                    print("Plotting Validation Batch")
                    self.best_val = val_loss.item()

            else:
                return

            self.logger.experiment.add_figure('z_proj', plot_latents(outputs[0], color_by="class_weight")['fig'], self.current_epoch)
            fig = self.plot_band_embeddings()
            if fig is not None:
                self.logger.experiment.add_figure("Normalized Band Embedding", fig, self.current_epoch)


            sqrt_num_axes = min(int(np.sqrt(len(outputs))), 2) # Only plot 4 samples because plots get huge
            for name in ("Best", "Worst", "Median", "Static"):
                fig, axes = plt.subplots(sqrt_num_axes, sqrt_num_axes, figsize=(6.4*sqrt_num_axes, 4.8*sqrt_num_axes))
                for i, ax in enumerate(axes.flat if hasattr(axes, '__iter__') else [axes]):
                    result = outputs[i]
                    loss_val = result["loss"]
                    out = result["out"]
                    batch = result["batch"]

                    squared_error = result['chi2'].detach().cpu().numpy()[:, 0]
                    if name == "Best":
                        idx = squared_error.argmin()
                    elif name == "Worst":
                        idx = squared_error.argmax()
                    elif name == "Median":
                        idx = np.argsort(squared_error)[len(squared_error)//2]
                    elif name == "Static":
                        idx = 0
                    else:
                        raise ValueError(f"{name} is an invalid lookup metric")

                    plot_reconstruction_from_batch(self, batch, idx_in_batch=idx, ax=ax, device=self.device, inverse_scale_reconstruction=self.inverse_scale_reconstruction)
                fig.tight_layout()
                self.logger.experiment.add_figure(name, fig, self.current_epoch)
        finally:
            self.validation_step_outputs.clear()

