import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from TAE.experiments.experiment import Experiment


class GRUVAEExperiment(Experiment):
    """
    PyTorch LightningModule for training the single-band EnhancedSNModel.
    Computes reconstruction loss, KL divergence, and smoothness penalty.
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

        loss = self.model.loss_function(
            recon=out["reconstructed"],
            target=batch["flux_full"],
            time=batch["time_full"],
            mask=batch["in_sample_mask"],
            z_mean=out["z_mean"],
            z_logvar=out["z_logvar"],
        )

        self.log("val_loss", loss["loss"])
        return loss, out, batch

    def validation_epoch_end(self, outputs):

        loss, out, batch = outputs[0]
        out_time = torch.linspace(-1, 1, 200, dtype=torch.float32)

        squared_error = (
            ((batch["flux_full"] - out["reconstructed"]) ** 2)
            .mean(axis=1)
            .detach()
            .cpu()
            .numpy()
        )
        best_idx = squared_error.argmin()
        worst_idx = squared_error.argmax()

        for idx, name in zip((best_idx, worst_idx), ("Best", "Worst")):
            fig = plt.figure()
            length = batch["lengths"][idx].detach().cpu().numpy()
            flux = batch["flux_full"][idx].detach().cpu().numpy()[:length, 0]
            flux_err = batch["flux_err_full"][idx].detach().cpu().numpy()[:length, 0]
            time = batch["time_full"][idx].detach().cpu().numpy()[:length, 0]
            band = batch["band_idx_full"][idx].detach().cpu().numpy()[:length, 0]
            colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
            for j in range(6):
                bands = torch.ones(200, dtype=torch.long).unsqueeze(1).unsqueeze(0) * j
                out = self.model(
                    x=batch["x_input"],
                    t=out_time.unsqueeze(1)
                    .unsqueeze(0)
                    .expand(len(batch["x_input"]), 200, 1)
                    .to(self.device),
                    band_idx_part=batch["band_idx"],
                    band_idx_full=bands.expand(len(batch["x_input"]), 200, 1).to(
                        self.device
                    ),  # Needs to be bands for decoding per time!
                )

                reconstruction = (
                    out["reconstructed"][idx].detach().cpu().numpy()[:, 0]
                )
                mask = band == j

                plt.errorbar(
                    time[mask],
                    flux[mask],
                    yerr=flux_err[mask],
                    marker="o",
                    label=f"band {j}",
                    ls="none",
                    color=colors[j],
                )
                plt.plot(out_time, reconstruction, label="Prediction", color=colors[j])
            plt.legend()
            plt.xlabel("Scaled Time")
            plt.ylabel("Scaled Flux")
            plt.xlim(-time.max() * 1.2, time.max() * 1.2)

            self.logger.experiment.add_figure(name, fig, self.current_epoch)
