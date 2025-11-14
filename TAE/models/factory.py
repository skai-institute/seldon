from TAE.models.decoders import Label_MLPDecoder, ConvDecoder, MLPBandDecoder


from TAE.models.encoders import Label_MLPEncoder, MLPEncoder, ConvEncoder, GRUMaskedEncoder
from TAE.models.deepset import DeepSetEncoder
from TAE.models.optimizer import Adam, RAdam, Ranger
from TAE.models.scheduler import (
    AnnealingLinearScheduler,
    ConstantScheduler,
    ConstrainedExponentialSchedulerMaLagrange,
)
from TAE.models.vae_label import VAELabel
from TAE.models.autoencoders import Autoencoder, BandAwareAutoencoder, StudentGRUMasked
from TAE.util import registry
from TAE.models.ModelNN import ResNet
from TAE.models.neuralode import RNNODEEncoder
from TAE.models.loss import *
from TAE.models.embeddings import *

# from TAE.experiments.autoencoder_experiment import Experiment, AutoencoderExperiment

from inspect import signature

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"

MODULES = {
    "MLPEncoder": MLPEncoder,
    "MLPBandDecoder": MLPBandDecoder,
    "AnnealingLinearScheduler": AnnealingLinearScheduler,
    "ConstrainedExponentialSchedulerMaLagrange": ConstrainedExponentialSchedulerMaLagrange,
    "RAdam": RAdam,
    "Ranger": Ranger,
    "Adam": Adam,
    "Label_MLPEncoder": Label_MLPEncoder,
    "Label_MLPDecoder": Label_MLPDecoder,
    "ConstantScheduler": ConstantScheduler,
    "ConvDecoder": ConvDecoder,
    "ConvEncoder": ConvEncoder,
    "VAELabel": VAELabel,
    "Autoencoder": Autoencoder,
    "BandAwareAutoencoder": BandAwareAutoencoder,
    "GRUMaskedEncoder": GRUMaskedEncoder,
    "student_gru_masked": StudentGRUMasked,
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
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
