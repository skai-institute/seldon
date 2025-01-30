from TAE.models.decoders import Label_MLPDecoder, MLPDecoder, ConvDecoder
from TAE.models.encoders import Label_MLPEncoder, MLPEncoder, ConvEncoder
from TAE.models.optimizer import Adam, RAdam, Ranger
from TAE.models.scheduler import (
    AnnealingLinearScheduler,
    ConstantScheduler,
    ConstrainedExponentialSchedulerMaLagrange,
)
from TAE.models.vae_label import VAELabel

MODULES = {
    "MLPEncoder": MLPEncoder,
    "MLPDecoder": MLPDecoder,
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
}

# TODO: switch it back when the cluster updated to python 3.8
# NAME_KEY: typing.Final[str] = "name"
# CONFIG_KEY: typing.Final[str] = "config"
NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config):
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
