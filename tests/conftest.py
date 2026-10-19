import os
import sys

# Must be set BEFORE any TAE import triggers matplotlib or torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure repo root is on sys.path so `import TAE` works regardless of
# where pytest is invoked from (CI runner cwd may differ)
repo_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

# Import TAE.util early to register OmegaConf resolvers ("sum", "len")
# This must happen once, before any config test calls OmegaConf.resolve()
import TAE.util  # noqa: F401

import pytest


@pytest.fixture
def repo_root():
    """Return the absolute path to the repository root."""
    return repo_root_path


@pytest.fixture
def data_dir(repo_root):
    """Return the absolute path to TAE/data/."""
    return os.path.join(repo_root, "TAE", "data")


@pytest.fixture
def configs_dir(repo_root):
    """Return the absolute path to TAE/configs/."""
    return os.path.join(repo_root, "TAE", "configs")
