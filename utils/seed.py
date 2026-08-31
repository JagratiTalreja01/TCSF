"""
seed.py

Utility functions for reproducibility.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")