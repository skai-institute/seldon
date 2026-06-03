import torch
import lightning.pytorch as pl
from TAE.experiments.experiment import Experiment

from TAE.models.metrics_results import MetricsEvaluator, Metrics
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.visualizations.latent_space import plot_latents
import numpy as np
import matplotlib.pyplot as plt
from TAE.datasets.lightcurve_dataset import enhanced_collate_fn
from TAE.visualizations.percentile_forecasting import percentile_forecast


class GRUVAEExperiment(Experiment):
    """
    LightningModule wrapper for training VAE-based SN models.
    Computes reconstruction loss, KL, smoothness. Logs visuals and metrics.
    """

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
            time_sorted=batch['time_sorted'],
            sequence_batch_mask=batch['sequence_batch_mask'],
        )

    def training_step(self, batch, batch_idx):
        results = self.model.training_step(batch, batch_idx)
        self.log("train_loss", results["loss"], sync_dist=True)
        self.log("train_recon", results['reconstruction_loss'], sync_dist=True)
        self.log("train_kl", results['kl_divergence'], sync_dist=True)
        return results

    def validation_step(self, batch, batch_idx):

        out = self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"],
            mask=batch["pad_mask_part"],
            time_sorted=batch['time_sorted'],
            sequence_batch_mask=batch['sequence_batch_mask'],
        )

        loss = self.model.loss_function(out, batch)
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
        if not hasattr(self, "best_val"):
            self.best_val = 1e32

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






        
