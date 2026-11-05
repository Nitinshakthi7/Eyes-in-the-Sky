"""utils.py - small helpers shared by all parts.

Owner: Nitin
Part:  D (shared train() + data-efficiency study + repo)

Contains:
  set_seed(seed)  - make runs repeatable
  get_device()    - GPU if PyTorch can use one, else CPU
  paths           - ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, SPLITS_CSV, METRICS_CSV

Check that it works (venv on, from the repo folder):  python -m src.utils
"""

import random
from pathlib import Path

import numpy as np
import torch

# ---------- paths ----------
# This file is src/utils.py. .parent is the src folder, .parent.parent is the repo folder.
# Everyone imports these names, so nobody types a path by hand (our folder name has spaces).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"                # saved weights (*.pth), not in git
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SPLITS_CSV = DATA_DIR / "splits.csv"        # the fixed 70/15/15 split (part A)
METRICS_CSV = RESULTS_DIR / "metrics.csv"   # one row per finished run (train() in part D)


def set_seed(seed):
    """Make a run repeatable: the same seed gives the same random numbers."""
    random.seed(seed)                          # Python's own random numbers
    np.random.seed(seed)                       # numpy's random numbers
    torch.manual_seed(seed)                    # PyTorch's random numbers (CPU and GPU)
    torch.backends.cudnn.deterministic = True  # the GPU uses the same convolution method every time
    torch.backends.cudnn.benchmark = False     # do not let the GPU try methods and pick by speed
    # note: two different laptops can still give slightly different numbers


def get_device():
    """Return the GPU if PyTorch can use one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Tiny self-check. Run it with:  python -m src.utils
    set_seed(0)
    a = torch.rand(3)
    set_seed(0)
    b = torch.rand(3)
    set_seed(1)
    c = torch.rand(3)
    assert torch.equal(a, b), "the same seed must give the same numbers"
    assert not torch.equal(a, c), "a different seed must give different numbers"
    for folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        assert folder.is_dir(), f"missing folder: {folder}"
    print("set_seed: OK (same seed gives the same numbers)")
    print("get_device:", get_device())
    print("paths: OK, repo folder is", ROOT)
