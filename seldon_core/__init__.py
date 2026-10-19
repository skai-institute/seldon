"""Alias package: ``seldon_core`` is the installable name for ``TAE``.

Downstream consumers (e.g. the ``seldon-dsb`` repository) import this
library as ``seldon_core``; inside this repository the package keeps its
historical name ``TAE``. Importing ``seldon_core`` eagerly imports the
``TAE`` package and registers every ``TAE`` submodule in ``sys.modules``
under the matching ``seldon_core`` name, so both import paths yield the
*same* module objects -- no duplicate classes, and ``isinstance`` checks
hold across the two names.

The walk is filesystem-based (not ``pkgutil``) because the ``TAE``
subpackages are PEP 420 namespace packages (no ``__init__.py``), which
``pkgutil.walk_packages`` does not descend into. Submodules that fail to
import (e.g. an optional dependency absent from the environment) are
skipped here; importing them via ``seldon_core.<name>`` then surfaces
the same error the ``TAE.<name>`` import would.
"""

import importlib
import logging
import os
import sys
from typing import Iterator

import TAE

_logger = logging.getLogger(__name__)
_ALIAS = "seldon_core"

# Replace this module with the real package, so `import seldon_core`
# and `import TAE` yield the same module object.
sys.modules[_ALIAS] = TAE


def _iter_module_names() -> Iterator[str]:
    """Yield dotted names of every TAE subpackage and submodule.

    Returns:
        Iterator over dotted module names (e.g. ``TAE.models.encoders``),
        parents before children. Hidden and ``__pycache__`` directories
        are skipped; data-only directories are yielded as namespace
        packages (importing them is harmless).
    """
    for root_path in TAE.__path__:
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith((".", "__"))
            )
            rel = os.path.relpath(dirpath, root_path)
            parts = [] if rel == "." else rel.split(os.sep)
            dotted = ".".join(["TAE"] + parts)
            if parts:
                yield dotted
            for filename in sorted(filenames):
                if filename.endswith(".py") and filename != "__init__.py":
                    yield dotted + "." + filename[: -len(".py")]


# Register every TAE submodule under its seldon_core name. sys.modules
# is checked first on dotted imports, so `import seldon_core.x.y`
# resolves to the already-imported TAE.x.y module object.
for _name in _iter_module_names():
    try:
        sys.modules[_ALIAS + _name[len("TAE") :]] = importlib.import_module(_name)
    except Exception as _exc:  # missing optional deps: skip, error on use
        _logger.debug("seldon_core alias skipped %s: %s", _name, _exc)
