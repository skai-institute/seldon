"""Tests for the ``seldon_core`` alias package (pip-installability).

The ``seldon_core`` package is a thin alias for ``TAE`` so that
downstream consumers (e.g. the ``seldon-dsb`` repository) can
``import seldon_core`` while this repository keeps the historical
``TAE`` package name. These tests assert that the two import paths
yield the *same* module objects -- the property that prevents
duplicate-class bugs (failed ``isinstance`` checks across the names).
"""

import sys

import TAE
import TAE.util


def test_root_alias_is_same_module() -> None:
    """``import seldon_core`` yields the TAE package object itself."""
    import seldon_core

    assert sys.modules["seldon_core"] is TAE
    assert seldon_core is TAE


def test_plain_submodule_identity() -> None:
    """A top-level submodule resolves to the same object via both names."""
    import seldon_core.util

    assert seldon_core.util is TAE.util


def test_namespace_subpackage_submodule_identity() -> None:
    """A module inside a namespace subpackage is shared, not duplicated."""
    import seldon_core.models.encoders
    import TAE.models.encoders

    assert (
        sys.modules["seldon_core.models.encoders"]
        is sys.modules["TAE.models.encoders"]
    )


def test_from_import_yields_identical_objects() -> None:
    """``from seldon_core... import X`` and the TAE form give the same X."""
    from seldon_core.models.factory import get_module as alias_get_module
    from TAE.models.factory import get_module as tae_get_module

    assert alias_get_module is tae_get_module
