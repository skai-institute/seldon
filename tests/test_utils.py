"""R1: Tests for utility functions in TAE/util.py."""

import os
import numpy as np
import torch

from TAE.util import (
    read_transmission_filter,
    map_transmission_filters,
    linear_to_log,
    log_to_linear,
    flatten_config,
)


class TestReadTransmissionFilter:
    def test_rubin_filter(self, data_dir):
        """Rubin/LSST filter: 'total' in filename triggers nm-to-Angstrom conversion."""
        path = os.path.join(data_dir, "total_g.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert isinstance(throughput, np.ndarray)
        assert len(wavelength) == len(throughput)
        assert len(wavelength) > 0
        # After nm * 10 conversion, wavelengths should be in Angstroms (3000-11000 range)
        assert wavelength.min() > 1000
        assert throughput.min() >= 0.0
        assert throughput.max() <= 1.0

    def test_jwst_filter(self, data_dir):
        """JWST filter: 'jwst' in filename triggers micron-to-Angstrom conversion."""
        path = os.path.join(data_dir, "jwst_nircam_f200w.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert len(wavelength) == len(throughput)
        # After micron * 1e4 conversion, wavelengths should be in Angstroms (> 1000)
        assert wavelength.min() > 1000

    def test_roman_filter(self, data_dir):
        """Roman filter: no unit conversion (neither 'jwst' nor 'total' in filename)."""
        path = os.path.join(data_dir, "Roman_WFI.F062.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert len(wavelength) == len(throughput)
        assert len(wavelength) > 0


class TestMapTransmissionFilters:
    def test_basic(self, data_dir):
        """Pass a dict of filter filenames, get back wave_grid and embeddings."""
        filters = {
            "g": os.path.join(data_dir, "total_g.dat"),
            "r": os.path.join(data_dir, "total_r.dat"),
        }
        wave_grid, filter_embeddings = map_transmission_filters(filters)

        assert isinstance(wave_grid, np.ndarray)
        assert len(wave_grid) > 0
        assert isinstance(filter_embeddings, dict)
        assert set(filter_embeddings.keys()) == {"g", "r"}
        for key, emb in filter_embeddings.items():
            assert len(emb) == len(wave_grid)


class TestLinearLogRoundtrip:
    def test_roundtrip(self):
        """log_to_linear(linear_to_log(x)) should recover x."""
        x = torch.tensor([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 100.0])
        recovered = log_to_linear(linear_to_log(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_zero(self):
        """linear_to_log of zero should be zero."""
        x = torch.tensor([0.0])
        result = linear_to_log(x)
        assert result.item() == 0.0


class TestFlattenConfig:
    def test_nested_dict(self):
        config = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        result = flatten_config(config)
        assert result == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_flat_dict(self):
        config = {"x": 1, "y": 2}
        result = flatten_config(config)
        assert result == {"x": 1, "y": 2}
