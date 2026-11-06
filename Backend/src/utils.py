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

# ---------- folder paths ----------
# We define all folder paths here in one place so no one has to type them by hand anywhere else.
# Path(__file__) gives the full path to this file (utils.py).
# .parent goes up one folder (into src/), .parent again goes up to the repo root.
# We do this because the repo folder name has spaces in it, which causes bugs if typed raw.
ROOT = Path(__file__).resolve().parent.parent   # the repo root folder (eyes-in-the-sky/)
DATA_DIR = ROOT / "data"                         # where the EuroSAT images and splits.csv live
MODELS_DIR = ROOT / "models"                     # where trained model weights (.pth files) are saved (not in git)
RESULTS_DIR = ROOT / "results"                   # where metrics.csv and misclassified image lists go
FIGURES_DIR = RESULTS_DIR / "figures"            # where all chart/plot images are saved
SPLITS_CSV = DATA_DIR / "splits.csv"             # the fixed 70/15/15 train/val/test split (made by Part A)
METRICS_CSV = RESULTS_DIR / "metrics.csv"        # one row per finished training run (written by train() in Part D)


def set_seed(seed):
    """Make a training run repeatable.

    When you give the same seed number, the model starts with the same random weights,
    sees images in the same order, and uses the same random augmentations.
    This means two people running the same seed on the same hardware get the same result.
    We call this at the start of every train() call.
    """
    random.seed(seed)                          # fixes Python's built-in random number generator
    np.random.seed(seed)                       # fixes numpy's random number generator
    torch.manual_seed(seed)                    # fixes PyTorch's random numbers on both CPU and GPU
    torch.backends.cudnn.deterministic = True  # tells the GPU to always use the same convolution algorithm
    torch.backends.cudnn.benchmark = False     # stops the GPU from auto-picking a faster algorithm (which would vary run to run)
    # NOTE: even with the same seed, two different machines can give slightly different numbers
    # because they use different floating-point hardware.


def get_device():
    """Return the GPU if available, otherwise the CPU.

    We use this everywhere so the whole project uses the GPU automatically
    without anyone having to change any code. If the machine has no NVIDIA GPU,
    training falls back to CPU (slower but still works).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Quick self-check. Run with:  python -m src.utils
    # This verifies that set_seed actually makes numbers repeatable,
    # and that all the expected folders exist on this machine.

    set_seed(0)
    a = torch.rand(3)
    set_seed(0)
    b = torch.rand(3)
    set_seed(1)
    c = torch.rand(3)

    # Same seed must give the same 3 random numbers
    assert torch.equal(a, b), "the same seed must give the same numbers"
    # Different seed must give different numbers
    assert not torch.equal(a, c), "a different seed must give different numbers"

    # Check all project folders exist (they should be created by check_setup.py)
    for folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        assert folder.is_dir(), f"missing folder: {folder}"

    print("set_seed: OK (same seed gives the same numbers)")
    print("get_device:", get_device())
    print("paths: OK, repo folder is", ROOT)
