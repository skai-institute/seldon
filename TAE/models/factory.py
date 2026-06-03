import TAE.models.decoders


import TAE.models.encoders
from TAE.models.deepset import DeepSetEncoder
from TAE.models.optimizer import Adam
from TAE.models.scheduler import (
    AnnealingLinearScheduler,
    ConstantScheduler,
    ConstrainedExponentialSchedulerMaLagrange,
    SquareRootWarmupScheduler
)
from TAE.models.vae_label import VAELabel
from TAE.models.autoencoders import Autoencoder, BandAwareAutoencoder, StudentGRUMasked
from TAE.util import registry
from TAE.models.ModelNN import *
from TAE.models.neuralode import RNNODEEncoder
from TAE.models.loss import *
from TAE.models.embeddings import *
from TAE.models.seldon import *
from TAE.energy import *

# from TAE.experiments.autoencoder_experiment import Experiment, AutoencoderExperiment

from inspect import signature

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"

MODULES = {
    "MLPBandDecoder": MLPBandDecoder,
    "AnnealingLinearScheduler": AnnealingLinearScheduler,
    "ConstrainedExponentialSchedulerMaLagrange": ConstrainedExponentialSchedulerMaLagrange,
    "Adam": Adam,
    "ConstantScheduler": ConstantScheduler,
    "Autoencoder": Autoencoder,
    "BandAwareAutoencoder": BandAwareAutoencoder,
    # "Experiment": Experiment,
    # "AutoencoderExperiment": AutoencoderExperiment,
}
MODULES.update(registry)


def get_module(name: str, config) -> object:
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )

    cls = MODULES[name]
    if config is None:
        return cls
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
