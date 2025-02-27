from TAE.experiments.experiment import Experiment


class AutoencoderExperiment(Experiment):

    def forward(self, x, mask):

        return self.model(x, mask)
    
    def training_step(self, batch, batch_idx):

        x_ts, mask, _, _ = batch
        x_ts = x_ts.view(x_ts.size(0), -1)
        mask = mask.view(mask.size(0), -1)
        self.curr_device = self.curr_device
        result = self(x_ts, mask)

        train_loss = self.model.loss_function(result)

        self.log_dict({f"train_{k}": v for k, v in train_loss.items()})

        return train_loss
    
    def validation_step(self, batch, batch_idx):

        x_ts, mask, _, _ = batch
        x_ts = x_ts.view(x_ts.size(0), -1)
        mask = mask.view(mask.size(0), -1)

        self.curr_device = self.curr_device
        result = self(x_ts, mask)

        val_loss = self.model.loss_function(result)

        self.log_dict({f"val_{k}": v for k, v in val_loss.items()})

        return []