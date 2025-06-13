import torch
import pytorch_lightning as pl
from TAE.experiments.experiment import Experiment

from TAE.models.metrics_results import MetricsEvaluator, Metrics
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.visualizations.latent_space import plot_latents
import numpy as np
import matplotlib.pyplot as plt


class GRUVAEExperiment(Experiment):
    """
    LightningModule wrapper for training VAE-based SN models.
    Computes reconstruction loss, KL, smoothness. Logs visuals and metrics.
    """

    def forward(self, batch):
        """
        Forward pass: reconstruct flux values over full time grid.
        """
        #print(batch['in_sample_mask_part'].shape, batch['x_input'].shape)
        return self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"],
            mask=batch["in_sample_mask_part"]
        )

    def training_step(self, batch, batch_idx):
        results = self.model.training_step(batch, batch_idx)
        self.log("train_loss", results["loss"])
        return results

    def validation_step(self, batch, batch_idx):
        
        out = self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"],
            mask=batch["in_sample_mask_part"]
        )
        
        loss = self.model.loss_function(out, batch)

        return {"loss": loss["loss"], "out": out, "batch": batch, 'chi2': loss['squared_error']}

    def validation_epoch_end(self, outputs):
        result = outputs[0]  # just visualize one batch
        loss_val = result["loss"]
        out = result["out"]
        batch = result["batch"]

        self.log("val_loss", torch.mean(torch.tensor([o['loss'] for o in outputs])))

        if self.metrics is not None:
            metrics = self.metrics(outputs[0])
            for metric in metrics:
                #self.log(metric, metrics[metric])
                self.logger.experiment.add_histogram(metric, metrics[metric], self.current_epoch)

        self.logger.experiment.add_figure('z_proj', plot_latents(outputs[0])['fig'], self.current_epoch)

        sqrt_num_axes = int(np.sqrt(len(outputs)))
        for name in ("Best", "Worst", "Median"):
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
                else:
                    raise ValueError(f"{name} is an invalid lookup metric")

                plot_reconstruction_from_batch(self.model, batch, idx_in_batch=idx, ax=ax, device=self.device)
            fig.tight_layout()
            self.logger.experiment.add_figure(name, fig, self.current_epoch)

        
