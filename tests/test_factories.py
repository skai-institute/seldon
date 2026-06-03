"""R2: Tests for factory registration in models, datasets, and experiments."""

import pytest


class TestModelFactory:
    def test_expected_keys(self):
        from TAE.models.factory import MODULES
        for key in ["Autoencoder", "BandAwareAutoencoder", "GRUMaskedEncoder", "Adam"]:
            assert key in MODULES, f"Expected '{key}' in models MODULES"

    def test_unknown_module_raises(self):
        from TAE.models.factory import get_module
        with pytest.raises(KeyError):
            get_module("nonexistent_module_xyz", {})


class TestDatasetFactory:
    def test_expected_keys(self):
        from TAE.datasets.dataset_factory import MODULES
        for key in ["YSEData", "SNDataModule"]:
            assert key in MODULES, f"Expected '{key}' in datasets MODULES"

    def test_unknown_module_raises(self):
        from TAE.datasets.dataset_factory import get_module
        with pytest.raises(KeyError):
            get_module("nonexistent_module_xyz", {})


class TestExperimentFactory:
    def test_expected_keys(self):
        from TAE.experiments.factory import MODULES
        for key in ["GRUVAEExperiment", "SELDONExperiment"]:
            assert key in MODULES, f"Expected '{key}' in experiments MODULES"

    def test_unknown_module_raises(self):
        from TAE.experiments.factory import get_module
        with pytest.raises(KeyError):
            get_module("nonexistent_module_xyz", {})


class TestRegistryIntegration:
    def test_registry_is_populated(self):
        from TAE.util import registry
        assert len(registry) > 0, "Expected @register_module to populate the registry"

    def test_registry_entries_in_some_factory(self):
        """Each registry entry should appear in at least one factory's MODULES."""
        from TAE.util import registry
        from TAE.models.factory import MODULES as model_modules
        from TAE.datasets.dataset_factory import MODULES as dataset_modules
        from TAE.experiments.factory import MODULES as experiment_modules
        all_modules = {**model_modules, **dataset_modules, **experiment_modules}
        for name in registry:
            assert name in all_modules, (
                f"Registry entry '{name}' not found in any factory MODULES"
            )
