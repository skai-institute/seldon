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
    """read_transmission_filter returns wavelengths in MICRONS.

    The loader normalizes each instrument's native units (LSST files in
    nm, JWST files in microns, Roman files in Angstroms) to a common
    micron scale. These tests pin the returned ranges to each band's
    physical extent so a unit-branch regression (off by 10x or 1e4)
    fails loudly for every instrument.
    """

    def test_rubin_filter(self, data_dir):
        """Rubin/LSST filter: file in nm; returned in microns."""
        path = os.path.join(data_dir, "total_g.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert isinstance(throughput, np.ndarray)
        assert len(wavelength) == len(throughput)
        assert len(wavelength) > 0
        # LSST throughput files span ~300-1150 nm -> ~0.30-1.15 microns
        assert 0.2 < wavelength.min() < 0.5
        assert 1.0 < wavelength.max() < 1.5
        assert throughput.min() >= 0.0
        assert throughput.max() <= 1.0

    def test_jwst_filter(self, data_dir):
        """JWST filter: file natively in microns; returned in microns."""
        path = os.path.join(data_dir, "jwst_nircam_f200w.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert len(wavelength) == len(throughput)
        # F200W spans ~1.69-2.30 microns
        assert 1.5 < wavelength.min() < 2.0
        assert 2.0 < wavelength.max() < 2.5

    def test_roman_filter(self, data_dir):
        """Roman filter: file in Angstroms; returned in microns."""
        path = os.path.join(data_dir, "Roman_WFI.F062.dat")
        wavelength, throughput = read_transmission_filter(path)

        assert isinstance(wavelength, np.ndarray)
        assert len(wavelength) == len(throughput)
        assert len(wavelength) > 0
        # F062 spans ~0.46-0.80 microns; asserting the range guards the
        # cross-instrument unit consistency the loader exists to provide
        assert 0.3 < wavelength.min() < 0.6
        assert 0.6 < wavelength.max() < 1.0


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
