"""train_part_e.py - memory-efficient scratch CNN training for Part E experiments.

This is an alternative training script used specifically for Part E (seed 4 runs).
It uses the standard DataLoader approach (reads images from disk each epoch)
instead of the in-memory approach in train.py. This uses less RAM but is slower.

We use this when we want to run an extra seed (seed=4) on the scratch CNN for the
data-efficiency study without loading all 27,000 images into GPU memory at once.

Run from the Backend folder: python -m src.train_part_e
"""

import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.dataset import get_loader
from src.models import build_scratch_cnn
from src.train import _save_row
from src.utils import get_device, set_seed, METRICS_CSV

BATCH_SIZE = 64
EPOCHS = 20


def train_part_e(fraction, seed=4):
    """Train the scratch CNN baseline using the disk-based DataLoader.

    This mirrors the logic of train() in train.py but reads images from disk
    rather than from memory. It's slower per epoch but uses far less RAM.

    Parameters:
      fraction: share of training data to use (e.g. 0.25 = 25%)
      seed:     random seed for reproducibility (default 4, for the extra seed run)

    Flow per epoch:
      1. Training loop: forward pass → compute loss → backprop → update weights
      2. Validation loop: check val accuracy (no weight updates)
      3. If this is the best val accuracy so far, save the weights
    After all epochs: load best weights → score on test set → save to metrics.csv
    """
    device = get_device()
    set_seed(seed)

    # Build DataLoaders — these read images from disk on demand, batch by batch
    train_loader = get_loader(
        "train", fraction=fraction, seed=seed,
        augment=True, batch_size=BATCH_SIZE  # augmentation only on training data
    )
    val_loader = get_loader("val", batch_size=BATCH_SIZE)    # no augmentation
    test_loader = get_loader("test", batch_size=BATCH_SIZE)  # no augmentation

    # Build the scratch CNN and move it to GPU if available
    model = build_scratch_cnn().to(device)

    # Adam with the team-standard learning rate; foreach=False for reproducibility
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    # We track the best validation accuracy so we can restore the best weights at the end
    best_accuracy = -1
    best_weights = None
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        # --- Training phase ---
        model.train()  # enables dropout and batch statistics
        total_loss, correct, seen = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()       # clear old gradients
            scores = model(x)           # forward pass
            loss = loss_fn(scores, y)   # compute loss
            loss.backward()             # compute gradients
            optimizer.step()            # update weights

            total_loss += loss.item() * len(y)
            correct += (scores.argmax(1) == y).sum().item()
            seen += len(y)

        # --- Validation phase (no weight updates) ---
        model.eval()
        val_correct, val_seen = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                scores = model(x)
                val_loss += loss_fn(scores, y).item() * len(y)
                val_correct += (scores.argmax(1) == y).sum().item()
                val_seen += len(y)

        val_accuracy = val_correct / val_seen

        print(
            f"scratch_cnn | fraction {fraction:g} | seed {seed} | "
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {total_loss/seen:.4f}, "
            f"acc {correct/seen:.3f} | "
            f"val loss {val_loss/val_seen:.4f}, "
            f"acc {val_accuracy:.3f}"
        )

        # Save a copy of the weights if this is the best validation accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # Restore the best-epoch weights before scoring on the test set
    model.load_state_dict(best_weights)
    model.eval()

    # --- Final test scoring (done once, no decisions made from it) ---
    actual, predicted = [], []

    with torch.no_grad():
        for x, y in test_loader:
            scores = model(x.to(device))
            predicted.extend(scores.argmax(1).cpu().tolist())
            actual.extend(y.tolist())

    accuracy = np.mean(np.array(actual) == np.array(predicted))
    macro_f1 = f1_score(actual, predicted, average="macro")

    # Build the result row and save it to metrics.csv
    row = {
        "model": "scratch_cnn",
        "fraction": fraction,
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    _save_row(METRICS_CSV, row)

    seconds = time.perf_counter() - started
    print(
        f"\nTEST (best epoch {best_epoch}): "
        f"accuracy {accuracy:.4f}, macro-F1 {macro_f1:.4f} "
        f"| {seconds/60:.1f} min"
    )

    return row


if __name__ == "__main__":
    # Default: run on 25% of training data with seed 4
    train_part_e(fraction=0.25, seed=4)
