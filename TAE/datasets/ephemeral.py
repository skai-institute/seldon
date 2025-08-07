import json
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors import safe_open
from typing import Optional


# ---------- helpers ----------
_KEY_SEP = "/"  # how we join path components in flat safetensors keys

def _is_seq(x):
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

def _escape_key(k: str) -> str:
    """Escape path components so we can safely join with / (JSON Pointer style)."""
    s = str(k)
    return s.replace("~", "~0").replace("/", "~1")

def _to_numpy_leaf(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        arr = x.numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    elif np.isscalar(x):
        arr = np.array(x)
    else:
        raise TypeError(f"Unsupported leaf type: {type(x)}")
    if np.iscomplexobj(arr):
        raise TypeError("Complex dtypes not supported (set a split policy if needed).")
    return np.ascontiguousarray(arr)

# ---------- pack (flatten + save) ----------
def save(obj, path: str):
    """
    Convert a nested container to numpy leaves, flatten, and save to a .safetensors file.

    Leaves allowed: torch.Tensor, np.ndarray, Python scalars.
    Containers: dict (Mapping), list/tuple (Sequence).
    """
    flat: dict[str, np.ndarray] = {}

    def walk(o, prefix: Optional[str]):
        # Return a "structure node" mirroring the container, with leaves replaced by {"__leaf__": key}
        if isinstance(o, Mapping):
            return {k: walk(v, (_KEY_SEP.join(filter(None, [prefix, _escape_key(k)])) if prefix else _escape_key(k)))
                    for k, v in o.items()}
        if _is_seq(o):
            out = []
            for i, v in enumerate(o):
                key = (_KEY_SEP.join(filter(None, [prefix, str(i)])) if prefix else str(i))
                out.append(walk(v, key))
            return out
        # Leaf
        key = prefix or "value"
        flat[key] = _to_numpy_leaf(o)
        return {"__leaf__": key}

    structure = walk(obj, None)
    save_file(flat, path, metadata={"structure": json.dumps(structure), "key_sep": _KEY_SEP})

# ---------- load (read + unflatten) ----------
def load(path: str):
    """
    Load a previously saved nested structure (values are NumPy arrays).
    """
    with safe_open(path, framework="numpy") as f:
        meta = f.metadata() or {}
        structure = json.loads(meta["structure"])
        def unflatten(node):
            if isinstance(node, dict) and "__leaf__" in node:
                return f.get_tensor(node["__leaf__"])
            if isinstance(node, list):
                return [unflatten(x) for x in node]
            # dict node
            return {k: unflatten(v) for k, v in node.items()}
        return unflatten(structure)

# ---------------- usage ----------------
# data = {
#     "x": torch.randn(3, 4, device="cuda"),
#     "y": np.arange(10, dtype=np.int32),
#     "batch": [
#         {"a": torch.tensor([1., 2.])},
#         {"a": np.ones((2, 2), dtype=np.float32)}
#     ],
#     "scalar": 7,
# }
# save_nested_as_safetensors(data, "data.safetensors")
# restored = load_nested_from_safetensors("data.safetensors")  # same nesting, NumPy leaves

