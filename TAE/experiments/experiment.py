import typing

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import gamma

from TAE.models.factory import get_module
from TAE.models.metrics_results import MetricsEvaluator, Metrics

class Experiment(pl.LightningModule):
    def __init__(
        self,
        model: pl.LightningModule = None,
        params: typing.Dict[str, typing.Any] = None,
        data=None,
        metrics=None
    ) -> None:
        super(Experiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.data = data
        self.metrics = metrics
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except KeyError:
            pass

    def configure_optimizers(self):
        """optimizers. By default it is Adam. The options include RAdam, Ranger, and Adam_WU."""
        # self.params["optimizer"]["config"]["model"] = self.model
        optimizer_class = get_module(
            self.params["optimizer"]["name"],
            self.params["optimizer"]["config"],
        )
        optimizer = optimizer_class(self.model)
        return optimizer
