import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt


class GRUVAEExperiment(pl.LightningModule):
    """
    PyTorch LightningModule for training the single-band EnhancedSNModel.
    Computes reconstruction loss, KL divergence, and smoothness penalty.
    """

    def __init__(self, model, data, kl_weight=0.001, out_sample_lambda=0.8,
                 smooth_weight=1.0, lr=1e-3, **kwargs):
        """
        Args:
            model: the EnhancedSNModel (encoder + decoder)
            kl_weight: weight for KL divergence loss
            out_sample_lambda: weight for out-of-sample reconstruction loss
            smooth_weight: weight for smoothness penalty
            lr: learning rate
        """
        super().__init__()
        self.model = model
        self.data = data
        self.kl_weight = kl_weight
        self.out_sample_lambda = out_sample_lambda
        self.smooth_weight = smooth_weight
        self.lr = lr

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x_input, t_grid, band_idx_part, band_idx_full):
        """
        Forward pass: reconstruct flux values over full time grid.
        """
        return self.model(x_input, t_grid, band_idx_part, band_idx_full)


    def training_step(self, batch, batch_idx):
        out = self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"]
        )

        loss = self.compute_loss(
            recon=out["reconstructed"],
            target=batch["flux_full"],
            time=batch["time_full"],
            mask=batch["in_sample_mask"],
            z_mean=out["z_mean"],
            z_logvar=out["z_logvar"]
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        out = self.model(
            x=batch["x_input"],
            t=batch["time_full"],
            band_idx_part=batch["band_idx"],
            band_idx_full=batch["band_idx_full"]
        )

        loss = self.compute_loss(
            recon=out["reconstructed"],
            target=batch["flux_full"],
            time=batch["time_full"],
            mask=batch["in_sample_mask"],
            z_mean=out["z_mean"],
            z_logvar=out["z_logvar"]
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss, out, batch

    def compute_loss(self, recon, target, time, mask, z_mean, z_logvar):
        """
        Compute total loss = in-sample MSE + (out-of-sample MSE) + KL divergence + smoothness penalty.
        """
        squared_error = (recon - target) ** 2
        mask = mask.unsqueeze(-1)

        in_loss = (squared_error * mask).sum()
        out_loss = (squared_error * (~mask)).sum() * ((~mask).sum() + 1e-6) / (mask.sum() + 1e-6)
        kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # === Smoothness penalty ===
        #def compute_smoothness(y, t):
        #    dy = y[:, 1:] - y[:, :-1]
        #    dt = t[:, 1:] - t[:, :-1]
        #    d1 = dy / (dt + 1e-6)
        #    d2 = d1[:, 1:] - d1[:, :-1]
        #    dt2 = dt[:, 1:] + dt[:, :-1]
        #    return torch.mean((d2 / (dt2 + 1e-6)) ** 2)

        #smooth_penalty = compute_smoothness(recon, time)

        total_loss = (
            in_loss +
            self.out_sample_lambda * out_loss +
            self.kl_weight * kl)# +
         #   self.smooth_weight * smooth_penalty
        #)

        return total_loss

    def configure_optimizers(self):
        """
        Optimizer configuration.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_epoch_end(self, outputs):


        loss, out, batch = outputs[0]
        out_time = torch.linspace(0,1,100, dtype=torch.float32)
        
        
        squared_error = ((batch["flux_full"] - out["reconstructed"])**2).mean(axis=1).detach().cpu().numpy()
        best_idx = squared_error.argmin()
        worst_idx = squared_error.argmax()
        #out = self.model(
        #    x=batch["x_input"],
        #    t=out_time.unsqueeze(1).unsqueeze(0).expand(len(batch['x_input']), 100, 1).to(self.device),
        #    band_idx_part=batch["band_idx"],
        #    band_idx_full=batch["band_idx_full"] # Needs to be bands for decoding per time!
        #)

        for idx, name in zip((best_idx, worst_idx),("Best","Worst")):
            fig = plt.figure()
            length = batch["lengths"][idx].detach().cpu().numpy()
            flux = batch["flux_full"][idx].detach().cpu().numpy()[:length, 0]
            flux_err = batch["flux_err_full"][idx].detach().cpu().numpy()[:length, 0]
            time = batch["time_full"][idx].detach().cpu().numpy()[:length, 0]
            band = batch["band_idx_full"][idx].detach().cpu().numpy()[:length, 0]
            colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"]
            for j in range(6):
                bands = torch.ones(100, dtype=torch.long).unsqueeze(1).unsqueeze(0)*j
                out = self.model(
                    x=batch["x_input"],
                    t=out_time.unsqueeze(1).unsqueeze(0).expand(len(batch['x_input']), 100, 1).to(self.device),
                    band_idx_part=batch["band_idx"],
                    band_idx_full=bands.expand(len(batch['x_input']), 100, 1).to(self.device) # Needs to be bands for decoding per time!
                )


                reconstruction = out["reconstructed"][best_idx].detach().cpu().numpy()[:, 0]
                mask = band == j

                plt.errorbar(time[mask], flux[mask], yerr=flux_err[mask], marker='o', label=f'band {j}', ls='none', color=colors[j])
                plt.plot(out_time, reconstruction, label='Prediction', color=colors[j])
            plt.legend()
            plt.xlabel("Scaled Time")
            plt.ylabel("Scaled Flux")
            plt.xlim(0, time.max())

            self.logger.experiment.add_figure(name, fig, self.current_epoch)
