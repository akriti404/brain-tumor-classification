"""Seeding and device-selection utilities for reproducible experiments."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set every RNG we touch (python, numpy, torch, cuda) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where available; fall back silently otherwise
    # (some ops, e.g. certain interpolation kernels, have no deterministic impl).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device, honoring an explicit preference if given."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
