from inspect import signature

from TAE.experiments.experiment import (
    Experiment,
)

# from models.factory import MODULES as model_modules


MODULES = {
    "Experiment": Experiment,
}
# MODULES.update(model_modules)

# TODO: switch it back when the cluster updated to python 3.8
# NAME_KEY: typing.Final[str] = "name"
# CONFIG_KEY: typing.Final[str] = "config"
NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config) -> Experiment:
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    sig = signature(cls.__init__)
    for key, value in config.items():

        key_sig = sig.parameters[
            key
        ].annotation  # Check that we're not supposed to pass a dict!
        if hasattr(key_sig, "__origin__"):
            key_sig = key_sig.__origin__
        if isinstance(value, key_sig):
            pass

        elif isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
