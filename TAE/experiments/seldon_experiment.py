import torch
from torch import nn
import lightning.pytorch as pl
from TAE.experiments.experiment import Experiment
from TAE.models.loss import *
import typing
import torch.nn.functional as F

from TAE.models.metrics_results import MetricsEvaluator, Metrics
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.visualizations.latent_space import plot_latents
import numpy as np
import matplotlib.pyplot as plt
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
from TAE.visualizations.percentile_forecasting import percentile_forecast


class SELDONExperiment(Experiment):
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
                 loss = "mse",
                 group_latent_by_class: bool = False,
                 weight_bands=True,
                 weight_classes=True,
                 inverse_transform=True,
                 scale_delta_by=None,
                 tune_parameters=True,
                 *args, **kwargs):

        super().__init__(model, params, data, metrics)
        self.configure_loss(loss)
        self.kl_weight = kl_weight
        self.group_latent_by_class = group_latent_by_class
        self.weight_bands = weight_bands
        self.weight_classes = weight_classes
        self.inverse_transform = inverse_transform
        self.tune_parameters = tune_parameters
        self.scale_delta_by = scale_delta_by
        self.best_val = 1e32

        assert scale_delta_by is None or isinstance(scale_delta_by, str), f"scale_delta_by must be str or None but got {scale_delta_by}"


    def configure_loss(self, loss_):

        if isinstance(loss_, str):
            if loss_ == "Huber":
                self.recon_loss = FlexibleHuberLoss()
            else:
                raise ValueError(f"argument `loss` must be one of ('Huber') but got '{loss_}'")
        else:
            self.recon_loss = loss_

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
    
    def loss_function(self, out, batch):
        """
        Compute total loss = in-sample MSE + (out-of-sample MSE) + KL divergence + smoothness penalty.
        """

        delta_scaler = None
        if self.scale_delta_by is not None:
            delta_scaler = batch[self.scale_delta_by]

        input, target, sigma = out["reconstructed"], batch["flux_full"], batch["flux_err_full"]
        if self.inverse_transform:
            input, sigma_pred = self.data.dataset.inverse_transform(input, sigma)
            input = input + torch.randn_like(sigma_pred) * sigma_pred
            target, sigma_true = self.data.dataset.inverse_transform(target, sigma)
            sigma = sigma_true
            #if self.current_epoch > 100:
            #    sigma = sigma_pred
            #else:
             #   sigma = sigma_true
            #sigma = torch.sqrt(torch.square(sigma_pred) + torch.square(sigma_true))

        reconstruction_loss = self.recon_loss(input, target, sigma, delta_scaler) # (B, S, E)
        
        if self.weight_bands:
            reconstruction_loss *= batch["band_weight"][:, :, None]

        z_mean, z_var = out["z_mean"], out["z_var"]
        if self.group_latent_by_class:
            z_mean = self.__group_latent_by_class(out, batch)
        kl_loss = self.model.variational_component.kl_divergence_(z_mean, z_var) # (B, Z)

        if self.weight_classes:
            reconstruction_loss *= batch["class_weight"][:, None, None]
            kl_loss *= batch["class_weight"][:, None]

        # Mean over Padding Mask
        padding_mask = batch["pad_mask_full"].unsqueeze(-1)
        reconstruction_loss = (padding_mask * reconstruction_loss).sum(1) / (padding_mask.sum(1))

        total_loss = (
            reconstruction_loss.mean() + self.kl_weight * kl_loss.mean()
        )

        if self.tune_parameters:
            param_loss = self.parameter_loss(out['parameters'])
            total_loss += param_loss.mean()

        return {"loss": total_loss, 'squared_error': reconstruction_loss, 
                'kl_divergence': kl_loss.mean(), "reconstruction_loss":reconstruction_loss.mean()}

    def training_step(self, batch, batch_idx):
        out = self(batch)
        results= self.loss_function(out, batch)
        self.log("train_loss", results["loss"], sync_dist=True)
        self.log("train_recon", results['reconstruction_loss'], sync_dist=True)
        self.log("train_kl", results['kl_divergence'], sync_dist=True)
        return results

    def validation_step(self, batch, batch_idx):

        out = self(batch)

        loss = self.loss_function(out, batch)
        self.log("val_loss", loss['loss'], sync_dist=True)
        self.log("val_recon", loss['reconstruction_loss'], sync_dist=True)
        self.log("val_kl", loss['kl_divergence'], sync_dist=True)


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

                    plot_reconstruction_from_batch(self.model, batch, idx_in_batch=idx, ax=ax, device=self.device)
                fig.tight_layout()
                self.logger.experiment.add_figure(name, fig, self.current_epoch)
        finally:
            self.validation_step_outputs.clear()






        
