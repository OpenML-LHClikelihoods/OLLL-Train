from collections.abc import Mapping
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import os

class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights."""


def get_device() -> torch.device:
    """
    Returns the CUDA device safely on MIG or full GPUs.
    Uses the CUDA_VISIBLE_DEVICES environment variable to pick the allocated device.
    Falls back to CPU if no GPU is available.
    """
    # CUDA_VISIBLE_DEVICES may contain a single MIG instance ID or full GPU
    if "CUDA_VISIBLE_DEVICES" in os.environ and torch.cuda.is_available():
        # PyTorch sees only the allowed GPU(s)
        device = torch.device("cuda:0")
        print(f"[INFO] Using CUDA device {torch.cuda.get_device_name(device)}")
        return device
    elif torch.cuda.is_available():
        # No CUDA_VISIBLE_DEVICES restriction
        device = torch.device("cuda:0")
        print(f"[INFO] Using CUDA device {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("[INFO] No CUDA available, using CPU")
        return torch.device("cpu")


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(
        -1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :]
    )


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def frequency_check(step, every_n_steps, skip_initial=False):
    """Checks whether an action should be performed at a given step and frequency.

    Parameters
    ----------
    step : int
        Step number (one-indexed)
    every_n_steps : None or int
        Desired action frequency. None or 0 correspond to never executing the action.
    skip_initial : bool
        If True, frequency_check returns False at step 0.

    Returns
    -------
    decision : bool
        Whether the action should be executed.
    """

    if every_n_steps is None or every_n_steps == 0:
        return False

    if skip_initial and step == 0:
        return False

    return step % every_n_steps == 0

def cosine_warmup_scheduler(optimizer, warmup_steps, T_max, eta_min=0):
    """
    Cosine annealing LR with linear warmup.

    Args:
        optimizer: your optimizer
        warmup_steps: number of steps to linearly increase LR from 0 -> max_lr
        total_steps: total number of training steps
        eta_min: minimum LR at the end of cosine decay
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # linear warmup 0 -> 1
            return float(step) / float(max(1, warmup_steps))
        else:
            # cosine annealing from 1 -> eta_min/max_lr
            progress = float(step - warmup_steps) / float(max(1, T_max - warmup_steps))
            return eta_min / optimizer.defaults['lr'] + (1 - eta_min / optimizer.defaults['lr']) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
