"""R12: Smoke test that catches Lightning 2.x silent-skip of on_validation_epoch_end.

Validates the rewritten-hook pattern with a minimal LightningModule. If
Lightning silently stops calling on_validation_epoch_end, the
val_epoch_sentinel logged from inside the hook would not appear in
trainer.callback_metrics and this test fails loudly.
"""

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset


class _SmokeModule(pl.LightningModule):
    """Minimal LightningModule exercising the R3 rewritten-hook pattern."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.validation_step_outputs: list = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        result = {"loss": loss}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        try:
            self.log("val_epoch_sentinel", torch.tensor(1.0, device=self.device), on_epoch=True, sync_dist=True)
            assert len(self.validation_step_outputs) > 0, (
                "validation_step_outputs should be non-empty when "
                "on_validation_epoch_end runs"
            )
        finally:
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def _make_loader():
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_on_validation_epoch_end_runs_under_lightning_2x():
    """Regression gate: Lightning must call the renamed hook."""
    pl.seed_everything(0, workers=True)
    model = _SmokeModule()
    trainer = pl.Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, _make_loader(), _make_loader())

    assert "val_epoch_sentinel" in trainer.callback_metrics, (
        "on_validation_epoch_end did not run — Lightning 2.x silently skipped "
        "the renamed hook."
    )
    assert torch.isfinite(trainer.callback_metrics["val_epoch_sentinel"]), (
        "val_epoch_sentinel should be a finite tensor (we log 1.0)"
    )
