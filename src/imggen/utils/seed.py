"""Seeding. numpy/torch are optional so the core package imports without them."""

from __future__ import annotations

import hashlib
import os
import random


def derive_seed(base: int, *keys: object) -> int:
    """Stable 32-bit sub-seed, e.g. derive_seed(seed, "noise", rank).

    Uses sha256 (not hash()) so it is identical across processes and machines.
    """
    blob = ":".join(str(x) for x in (base, *keys)).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
