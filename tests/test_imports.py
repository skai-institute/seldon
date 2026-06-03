"""Import smoke tests — catch missing dependencies early with clear errors."""


def test_import_tae():
    import TAE  # noqa: F401


def test_import_models_factory():
    from TAE.models.factory import MODULES, get_module  # noqa: F401


def test_import_datasets_factory():
    from TAE.datasets.dataset_factory import MODULES, get_module  # noqa: F401


def test_import_experiments_factory():
    from TAE.experiments.factory import MODULES, get_module  # noqa: F401


def test_import_util():
    from TAE.util import (  # noqa: F401
        read_transmission_filter,
        map_transmission_filters,
        linear_to_log,
        log_to_linear,
        flatten_config,
        registry,
    )
