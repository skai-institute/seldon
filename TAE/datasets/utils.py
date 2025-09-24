from collections.abc import Mapping, Sequence
import numpy as np
import torch

# Mapping of band names to integer codes
BAND_IDX_MAP = {"g": 0, "r": 1, "i": 2, "z": 3, "y": 4, "u": 5}

def _to_list(x):
    if x is None:
        return []
    return x if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)) else [x]

def _dataset_len(dl):
    ds = getattr(dl, "dataset", None)
    return len(ds) if ds is not None and hasattr(ds, "__len__") else None

def _sampler_len(dl):
    # Prefer explicit sampler lengths when present (works with DistributedSampler/WeightedRandomSampler/etc.)
    s = getattr(dl, "sampler", None)
    if s is not None:
        if hasattr(s, "num_samples"):
            return getattr(s, "num_samples")
        if hasattr(s, "__len__"):
            return len(s)
    bs = getattr(dl, "batch_sampler", None)
    if bs is not None and hasattr(bs, "__len__"):
        # Number of batches; cannot know last batch sizes without iterating.
        # If each batch is a fixed-size list/tuple, DataLoader may also set dl.batch_size.
        nb = len(bs)
        bs_val = getattr(dl, "batch_size", None)
        return nb * bs_val if bs_val is not None else None
    return None

def _per_epoch_samples_local(dl):
    """
    Best effort count of *examples* this DataLoader will yield per epoch on THIS rank.
    Preference order:
      1) sampler.num_samples / len(sampler)
      2) dataset length (when sampler is default non-replacement)
      3) len(dl) * batch_size (approx; wrong if last batch partial or custom batch_sampler)
    """
    n = _sampler_len(dl)
    if n is not None:
        return int(n)

    ds_len = _dataset_len(dl)
    if ds_len is not None:
        return int(ds_len)

    # Fallback approximation
    nb = len(dl) if hasattr(dl, "__len__") else None
    bs = getattr(dl, "batch_size", None)
    if nb is not None and bs is not None:
        return int(nb * bs)  # may overcount/undercount last batch depending on drop_last
    return None

def _describe_dl(dl, world_size=1):
    ds_len = _dataset_len(dl)
    local = _per_epoch_samples_local(dl)
    global_ = local * world_size if (local is not None and world_size is not None) else None
    return {
        "dataset_len": ds_len,            # items in the underlying dataset (if known)
        "per_epoch_local": local,         # samples drawn by THIS loader per epoch
        "per_epoch_global": global_,      # multiply by world_size (for DDP)
        "batch_size": getattr(dl, "batch_size", None),
        "drop_last": getattr(dl, "drop_last", None),
        "sampler": type(getattr(dl, "sampler", None)).__name__
                   if getattr(dl, "sampler", None) is not None else None,
    }

def datamodule_counts(dm, world_size=1, setup_stage="fit"):
    """
    Return counts for train/val/test/predict loaders in a LightningDataModule.
    Call dm.prepare_data()/dm.setup() as needed before.
    """
    # Ensure datasets/dataloaders exist
    if hasattr(dm, "prepare_data"):
        try:
            dm.prepare_data()
        except Exception:
            pass
    if hasattr(dm, "setup"):
        try:
            dm.setup(setup_stage)
        except Exception:
            # some DMs require stage=None to create everything
            try:
                dm.setup(None)
            except Exception:
                pass

    out = {}

    # train
    if hasattr(dm, "train_dataloader"):
        dls = _to_list(dm.train_dataloader())
        out["train"] = [_describe_dl(dl, world_size) for dl in dls]

    # val
    if hasattr(dm, "val_dataloader"):
        dls = _to_list(dm.val_dataloader())
        out["val"] = [_describe_dl(dl, world_size) for dl in dls]

    # test
    if hasattr(dm, "test_dataloader"):
        dls = _to_list(dm.test_dataloader())
        out["test"] = [_describe_dl(dl, world_size) for dl in dls]

    # predict
    # if hasattr(dm, "predict_dataloader"):
    #     dls = _to_list(dm.predict_dataloader())
    #     out["predict"] = [_describe_dl(dl, world_size) for dl in dls]

    # Totals (sum where known)
    def _sum_known(key):
        s = 0
        any_found = False
        for split in out.values():
            for d in split:
                if d[key] is not None:
                    s += int(d[key])
                    any_found = True
        return s if any_found else None

    out["_totals"] = {
        "dataset_len": _sum_known("dataset_len"),
        "per_epoch_local": _sum_known("per_epoch_local"),
        "per_epoch_global": _sum_known("per_epoch_global"),
    }
    return out


def to_cuda(obj, *, dtype=None, device=None, non_blocking=False):
    """
    Recursively walk through `obj`, convert every leaf to a `torch.Tensor`,
    and move the result to GPU.

    Parameters
    ----------
    obj : any
        A possibly-nested structure of lists/tuples/dicts whose leaves are
        scalars, NumPy arrays, or Tensors.
    dtype : torch.dtype, optional
        If given, every tensor is cast to this dtype before being moved.
    device : int | torch.device | None
        CUDA device to move to (e.g. 0 for "cuda:0").  Defaults to current
        CUDA device.
    non_blocking : bool, default False
        Passed straight through to `.cuda()` / `.to()` for asynchronous copies.

    Returns
    -------
    Same container type as `obj`, but with every leaf a CUDA tensor.
    """

    if isinstance(obj, str):
        # Do nothing to strings
        return obj

    if isinstance(obj, torch.Tensor):
        if dtype is not None:
            obj = obj.to(dtype)
        return obj.to(device=device or torch.cuda.current_device(),
                      non_blocking=non_blocking)

    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, dtype=dtype).to(
            device=device or torch.cuda.current_device(),
            non_blocking=non_blocking)

    # Handle mappings (dict, OrderedDict, …)
    if isinstance(obj, Mapping):
        return {k: to_cuda(v, dtype=dtype, device=device,
                           non_blocking=non_blocking)
                for k, v in obj.items()}

    # Handle sequences (list, tuple, range, …) but NOT strings/bytes
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        seq_type = type(obj)        # preserve list vs tuple
        return seq_type(to_cuda(v, dtype=dtype, device=device,
                                non_blocking=non_blocking) for v in obj)

    # Fallback: treat as scalar
    return torch.tensor(obj, dtype=dtype).to(
        device=device or torch.cuda.current_device(),
        non_blocking=non_blocking)
