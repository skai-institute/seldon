"""R4: Smoke tests for model instantiation on CPU."""

import torch


def test_autoencoder_instantiation():
    from TAE.models.autoencoders import Autoencoder
    model = Autoencoder(input_dim=100)
    assert isinstance(model, torch.nn.Module)


def test_autoencoder_has_parameters():
    from TAE.models.autoencoders import Autoencoder
    model = Autoencoder(input_dim=100)
    params = list(model.parameters())
    assert len(params) > 0


def test_variational_component_instantiation():
    from TAE.models.ModelNN import VariationalComponent
    model = VariationalComponent(input_dim=32, output_dim=8)
    assert isinstance(model, torch.nn.Module)


def test_model_factory_instantiates_autoencoder():
    from TAE.models.factory import get_module
    model = get_module("Autoencoder", {"input_dim": 64})
    assert isinstance(model, torch.nn.Module)
