"""
Random seed utility — ensures full reproducibility.
Call set_seed(seed) before any data loading or model creation.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fixes all sources of randomness for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
