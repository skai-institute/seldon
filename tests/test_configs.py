"""R3: Tests that all top-level Hydra configs load and resolve without errors."""

import os
import pytest
from omegaconf import OmegaConf

TOP_LEVEL_CONFIGS = [
    "SELDON_1.0.yaml",
    "SELDON_2.0.yaml",
    "SELDON_2.0_E_UQ_C.yaml",
    "SELDON_2.0_S_E_UQ_C.yaml",
    "SELDON_2.0_S_E_UQ_C_with_transmission.yaml",
    "SELDON_3.0.yaml",
    "light_curve_autoencoder.yaml",
    "light_curve_autoencoder_EQ.yaml",
    "light_curve_autoencoder_GRU.yaml",
    "light_curve_autoencoder_GRUODE.yaml",
    "light_curve_autoencoder_PureDeepSets.yaml",
    "student_GRU.yaml",
    "student_GRU_noKD.yaml",
    "teacher_GRU.yaml",
]

# Sections that every config must have
MINIMUM_SECTIONS = [
    "model_params",
    "dataset",
    "trainer_params",
    "logging_params",
]


@pytest.mark.parametrize("config_file", TOP_LEVEL_CONFIGS)
def test_config_loads_and_resolves(configs_dir, config_file):
    """Each top-level config should parse and resolve all interpolations."""
    path = os.path.join(configs_dir, config_file)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    for section in MINIMUM_SECTIONS:
        assert section in cfg, f"Missing section '{section}' in {config_file}"
