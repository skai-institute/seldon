import pytorch_lightning as pl
import torch



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

    def forward(self, x_input, t_grid, band_idx_full):
        """
        Forward pass: reconstruct flux values over full time grid.
        """
        return self.model(x_input, t_grid, band_idx_full)


    def training_step(self, batch, batch_idx):
        out = self.model(
            x=batch["x_input"],
            t=batch["time_full"],
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
        return loss

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
        def compute_smoothness(y, t):
            dy = y[:, 1:] - y[:, :-1]
            dt = t[:, 1:] - t[:, :-1]
            d1 = dy / (dt + 1e-6)
            d2 = d1[:, 1:] - d1[:, :-1]
            dt2 = dt[:, 1:] + dt[:, :-1]
            return torch.mean((d2 / (dt2 + 1e-6)) ** 2)

        smooth_penalty = compute_smoothness(recon, time)

        total_loss = (
            in_loss +
            self.out_sample_lambda * out_loss +
            self.kl_weight * kl +
            self.smooth_weight * smooth_penalty
        )

        return total_loss

    def configure_optimizers(self):
        """
        Optimizer configuration.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
