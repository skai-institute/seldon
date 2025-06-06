import torch
import pytorch_lightning as pl
from TAE.experiments.experiment import Experiment

from TAE.models.metrics_results import MetricsEvaluator, Metrics
from TAE.visualizations.reconstruction import plot_reconstruction_from_batch
from TAE.visualizations.latent_space import plot_latents


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
        )
        
        loss = self.model.loss_function(out, batch)
        self.log("val_loss", loss["loss"])

        return {"loss": loss["loss"], "out": out, "batch": batch, 'chi2': loss['squared_error']}

    def validation_epoch_end(self, outputs):
        result = outputs[0]  # just visualize one batch
        loss_val = result["loss"]
        out = result["out"]
        batch = result["batch"]

        #self.log("val_loss", loss_val)

        # Optional: log metrics like MeanFE or NRMSE
        #evaluator = MetricsEvaluator(self, batch)
        #metrics = evaluator.all_metrics()
        if self.metrics is not None:
            metrics = self.metrics(outputs[0])
            for metric in metrics:
                #self.log(metric, metrics[metric])
                self.logger.experiment.add_histogram(metric, metrics[metric], self.current_epoch)
        #self.log("val_meanFE", metrics["MeanFE"].mean())
        #self.log("val_NRMSE", metrics["NRMSE"].mean())

        # Select best and worst reconstructions by squared error
        squared_error = (
            ((batch["flux_full"] - out["reconstructed"]) ** 2)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        best_idx = squared_error.argmin()
        worst_idx = squared_error.argmax()

        for idx, name in zip((best_idx, worst_idx), ("Best", "Worst")):
            fig = plot_reconstruction_from_batch(self.model, batch, idx_in_batch=idx, device=self.device)
            self.logger.experiment.add_figure(name, fig, self.current_epoch)

        self.logger.experiment.add_figure('z_proj', plot_latents(outputs[0])['fig'], self.current_epoch)
