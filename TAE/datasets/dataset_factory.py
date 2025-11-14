import typing

from TAE.datasets.YSE_json_dataset import JSONDataLoader

# from TAE.datasets.datasets import RandomData
from TAE.datasets.lightcurve_dataset import SNDataModule
from TAE.datasets.lightcurve_dataset_parquet import SNDataModuleParquet
from TAE.datasets.lightcurve_dataloader import LightCurveDataLoader
from TAE.util import extract_curves_from_filtered_data
import pickle
from pathlib import Path
from TAE.util import registry

MODULES = {
    "YSEData": JSONDataLoader,
    "SNDataModule": SNDataModule,
}

MODULES.update(registry)

# TODO: switch it back when the cluster updated to python 3.8
# NAME_KEY: typing.Final[str] = "name"
# CONFIG_KEY: typing.Final[str] = "config"
NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


# In dataset_factory.py, modify the get_module function:
def get_module(name, config):
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
