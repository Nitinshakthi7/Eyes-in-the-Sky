"""train.py - the one shared training function used by all experiments.

Owner: Nitin
Part:  D (shared train() + data-efficiency study + repo)

Contains:
  train(model_name, fraction, seed)  - train one model with the team settings, add ONE row to results/metrics.csv
  build_model(model_name)            - returns the right model and its learning rate for a given name
  SCRATCH, TRANSFER                  - dictionaries of all model names train() understands

Team settings (docs/plan.md, section 4): batch 64, Adam, up to 20 epochs, keep the epoch with the best
validation accuracy. Decisions use the validation set only; the test set is scored ONCE, at the end of a run.

Speed: all 27,000 images are read from disk once and kept in memory (about 330 MB). Rotations, flips and
normalisation then happen on the GPU. It is the same maths as the loader in dataset.py, only much faster.

Check that it works (venv on, from the repo folder):  python -m src.train
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

from src import models
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, get_loader
from src.utils import DATA_DIR, METRICS_CSV, MODELS_DIR, SPLITS_CSV, get_device, set_seed

# Team standards: everyone uses the same batch size and max number of epochs
BATCH_SIZE = 64
MAX_EPOCHS = 20

# --- Scratch CNN model names ---
# Each entry maps a model name (string) to the keyword arguments for build_scratch_cnn().
# This way train() can build any scratch model just from its name string.
SCRATCH = {
    "scratch_cnn": {"use_batchnorm": True, "depth": 3},                # the full baseline model
    "scratch_cnn_no_batchnorm": {"use_batchnorm": False, "depth": 3},  # ablation: remove BatchNorm
    "scratch_cnn_depth2": {"use_batchnorm": True, "depth": 2},         # ablation: only 2 conv blocks
}

# --- Transfer learning model names ---
# Each entry maps a name to (backbone, freeze?).
# freeze=True means only the final layer trains; freeze=False fine-tunes everything.
TRANSFER = {
    "resnet18_frozen": ("resnet18", True),
    "resnet18_full": ("resnet18", False),
    "mobilenet_v2_frozen": ("mobilenet_v2", True),
    "mobilenet_v2_full": ("mobilenet_v2", False),
}


def build_model(model_name):
    """Return (model, learning_rate) for a given model name string.

    We use a lower learning rate (0.0001) when fine-tuning all layers of a pre-trained model
    because those weights are already close to good — large updates would destroy them.
    When only the final layer trains (frozen backbone), we can use a higher rate (0.001)
    because only a small fresh layer is learning from scratch.
    """
    # Build a scratch CNN if the name matches one of our ablations
    if model_name in SCRATCH:
        return models.build_scratch_cnn(**SCRATCH[model_name]), 1e-3

    # Build a transfer model if the name matches
    if model_name in TRANSFER:
        if not hasattr(models, "build_transfer_model"):
            raise NotImplementedError("build_transfer_model is not in src/models.py yet (Rachana, part C)")
        backbone, freeze = TRANSFER[model_name]
        # 0.001 for frozen (only the head trains), 0.0001 for full fine-tuning
        return models.build_transfer_model(backbone, freeze), (1e-3 if freeze else 1e-4)

    raise ValueError(f"unknown model name {model_name!r}; use one of {sorted([*SCRATCH, *TRANSFER])}")


# ---------- fast in-memory data loading ----------
# Instead of reading each image from disk every epoch (slow),
# we load all 27,000 images into RAM once at the start,
# then copy them to the GPU. Every epoch after that is much faster.
_cache = {}  # stores (images, labels, row_of_path) keyed by device string; filled once per session


def _load_all_images(device):
    """Read all 27,000 EuroSAT images into memory and move them to the GPU.

    We store them as raw byte tensors (values 0–255), NOT normalised yet.
    Normalisation and augmentation happen on-the-fly per batch inside _prepare().
    Storing as bytes is 4x more compact than storing as float32.

    Returns:
      images:       tensor of shape (27000, 3, 64, 64), dtype=uint8, on the device
      labels:       tensor of shape (27000,), dtype=int64, on the device
      row_of_path:  dict mapping a file path string to its row index in the tensor
    """
    key = str(device)
    if key not in _cache:
        df = pd.read_csv(SPLITS_CSV)
        print(f"Reading {len(df)} images into memory (one time, about 20 seconds)...")

        # Open every image as a numpy array and stack them all into one big array
        pictures = [np.array(Image.open(DATA_DIR / path).convert("RGB")) for path in df["path"]]

        # Stack: (27000, 64, 64, 3) → permute channels → (27000, 3, 64, 64)
        # contiguous() ensures the memory layout is correct for GPU operations
        images = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2).contiguous()

        labels = torch.tensor(df["label"].values)

        # Build a lookup dict: file path → row number, so we can find any image fast
        row_of_path = {path: row for row, path in enumerate(df["path"])}

        # Move everything to the GPU (or keep on CPU if no GPU)
        _cache[key] = (images.to(device), labels.to(device), row_of_path)

    return _cache[key]


def _rows(split, fraction, seed, row_of_path, device):
    """Return the row indices (into the big image tensor) for one split.

    We reuse get_loader() just to get the list of file paths for this split/fraction —
    no images are actually loaded from disk here, only splits.csv is read.
    Then we look up each path's row number in our in-memory tensor.
    """
    # get_loader with no images: we just want its .dataset.df to know which paths are in this split
    frame = get_loader(split, fraction=fraction if split == "train" else 1.0, seed=seed).dataset.df
    return torch.tensor([row_of_path[path] for path in frame["path"]], device=device)


def _prepare(batch_bytes, augment):
    """Convert a batch of raw byte images to normalised float tensors ready for the model.

    Steps:
    1. Divide by 255 to get float values in [0.0, 1.0]
    2. If augmenting: apply random 90-degree rotations and flips on the GPU (very fast)
    3. Subtract the ImageNet mean and divide by std — this centres the data around 0
       which is important for fast, stable training

    batch_bytes: tensor of shape (N, 3, H, W), dtype=uint8, values 0-255
    augment:     True during training, False during val/test
    Returns:     float tensor of same shape, normalised
    """
    x = batch_bytes.float() / 255.0  # scale to 0.0–1.0

    if augment:
        n = x.shape[0]

        # Pick a random rotation (0°, 90°, 180°, or 270°) for each image in the batch
        turns = torch.randint(0, 4, (n,), device=x.device)
        for k in (1, 2, 3):
            chosen = turns == k
            if chosen.any():
                x[chosen] = torch.rot90(x[chosen], k, dims=(2, 3))

        # Randomly flip left-right for 50% of images
        flip_sideways = torch.rand(n, device=x.device) < 0.5
        # Randomly flip upside-down for 50% of images
        flip_upside_down = torch.rand(n, device=x.device) < 0.5
        x[flip_sideways] = x[flip_sideways].flip(3)
        x[flip_upside_down] = x[flip_upside_down].flip(2)

    # Normalise: subtract mean, divide by std for each of the 3 colour channels
    # view(1,3,1,1) makes the shape broadcast correctly across the batch and spatial dims
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _score(model, rows, images, labels, loss_function):
    """Measure loss, accuracy and predictions on a set of rows, without augmentation.

    We use this for validation (after each epoch) and for the final test.
    model.eval() turns off dropout and makes BatchNorm use its running stats instead of
    the current batch stats — this is important for reproducible evaluation.
    torch.no_grad() tells PyTorch not to track gradients, saving memory and time.

    Returns: (average loss, accuracy as float 0–1, predictions tensor)
    """
    model.eval()
    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        # Process in chunks of 256 so we don't run out of GPU memory
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            scores = model(_prepare(images[batch], augment=False))
            total_loss += loss_function(scores, labels[batch]).item() * len(batch)
            predictions.append(scores.argmax(dim=1))  # pick the highest-scoring class

    predictions = torch.cat(predictions)
    accuracy = (predictions == labels[rows]).float().mean().item()
    return total_loss / len(rows), accuracy, predictions


def _save_row(path, row):
    """Add one result row to the metrics CSV file.

    If the same (model, fraction, seed) combination was run before, the old row
    is replaced rather than duplicated. This makes it safe to re-run experiments.

    row: a dict with keys: model, fraction, seed, accuracy, macro_f1
    """
    columns = ["model", "fraction", "seed", "accuracy", "macro_f1"]
    path = Path(path)

    # Read existing results, or start with an empty table if the file doesn't exist yet
    old = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=columns)

    if old.empty:
        new = pd.DataFrame([row], columns=columns)
    else:
        # Find rows that match this run's (model, fraction, seed) — they are the ones to replace
        same_run = ((old["model"] == row["model"]) & np.isclose(old["fraction"].astype(float), row["fraction"])
                    & (old["seed"] == row["seed"]))
        new = pd.concat([old.loc[~same_run], pd.DataFrame([row], columns=columns)], ignore_index=True)

    new.to_csv(path, index=False, lineterminator="\n")


def train(model_name, fraction, seed, epochs=MAX_EPOCHS, metrics_csv=METRICS_CSV, save_weights=None):
    """Train one model and record the final test accuracy in results/metrics.csv.

    This is the main function everyone calls. It handles everything:
    loading data, building the model, the training loop, early stopping (best val epoch),
    final test scoring, saving weights, and logging.

    Parameters:
      model_name:   a string from SCRATCH or TRANSFER (e.g. "scratch_cnn", "resnet18_full")
      fraction:     what fraction of the TRAINING data to use (0.05 to 1.0).
                    Val and test sets are always full-size regardless.
                    Used for the data-efficiency study (how much data do we need?).
      seed:         an integer — same seed = same starting weights, same image order, same augmentation
      epochs:       how many passes through the training data (default 20)
      metrics_csv:  where to write the result row (default: results/metrics.csv)
      save_weights: whether to save the best-epoch model weights to disk.
                    Default: only save when fraction=1.0 (the full training run).

    Returns a dict with accuracy, macro_f1, best_epoch, val_accuracy, seconds_per_epoch,
    and history (loss and accuracy curves per epoch).
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be above 0 and at most 1")

    # Set up: pick GPU or CPU, fix all random numbers for reproducibility
    device = get_device()
    set_seed(seed)

    # Load all images into memory once (or reuse from cache if already loaded)
    images, labels, row_of_path = _load_all_images(device)

    # Get the row indices for each split from the in-memory tensor
    train_rows = _rows("train", fraction, seed, row_of_path, device)
    val_rows = _rows("val", 1.0, seed, row_of_path, device)
    test_rows = _rows("test", 1.0, seed, row_of_path, device)

    # Build the model and move it to GPU
    model, learning_rate = build_model(model_name)
    model = model.to(device)

    # CrossEntropyLoss: standard loss for multi-class classification
    # It measures how wrong the model's probability predictions are
    loss_function = torch.nn.CrossEntropyLoss()

    # Adam optimiser: updates weights to reduce the loss each step.
    # We only pass parameters that are allowed to update (requires_grad=True).
    # This matters for frozen transfer models — we don't want to accidentally unfreeze them.
    # foreach=False matches what Tejash used in his runs (for reproducibility).
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, foreach=False)

    # This generator makes the image shuffle order the same every run with the same seed
    shuffler = torch.Generator().manual_seed(seed)

    # Tracking variables for training progress and model selection
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy, best_epoch, best_weights = -1.0, 0, None
    started = time.perf_counter()

    # ---- Main training loop ----
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()  # turn on training mode (enables dropout, batch stats for BN)

        # Accumulators for this epoch's stats
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0

        # Shuffle the training row indices for this epoch
        order = train_rows[torch.randperm(len(train_rows), generator=shuffler).to(device)]

        # Process images in batches of 64
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]

            # BatchNorm crashes if a batch has only 1 image, so we skip those edge cases
            if len(batch) < 2:
                continue

            # Prepare the batch: convert bytes → float → augment → normalise
            x = _prepare(images[batch], augment=True)
            y = labels[batch]

            # Standard PyTorch training step:
            optimizer.zero_grad()       # clear gradients from the previous step
            scores = model(x)           # forward pass: get class scores
            loss = loss_function(scores, y)  # compute how wrong we were
            loss.backward()             # backprop: compute how to fix each weight
            optimizer.step()            # update all weights by one step

            # Track loss and accuracy for this epoch's summary
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(dim=1) == y).sum()
            seen += len(batch)

        # --- Validation after each epoch ---
        # We use val_rows (never test_rows) to decide which epoch to keep.
        val_loss, val_accuracy, _ = _score(model, val_rows, images, labels, loss_function)
        history["train_loss"].append(total_loss.item() / seen)
        history["train_accuracy"].append(correct.item() / seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"{model_name} | fraction {fraction:g} | seed {seed} | epoch {epoch:02d}/{epochs} | "
              f"train loss {history['train_loss'][-1]:.4f}, acc {history['train_accuracy'][-1]:.3f} | "
              f"val loss {val_loss:.4f}, acc {val_accuracy:.3f} | {time.perf_counter() - epoch_started:.1f} s")

        # Keep a copy of the weights if this is the best validation accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            # We copy weights to CPU so they don't take GPU memory between epochs
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    # Restore the best-epoch weights before scoring on the test set
    model.load_state_dict(best_weights)

    # --- Final test scoring ---
    # This is the ONLY place we touch the test set.
    # We report these numbers, but make NO decisions based on them.
    _, accuracy, predictions = _score(model, test_rows, images, labels, loss_function)
    macro_f1 = f1_score(labels[test_rows].cpu().numpy(), predictions.cpu().numpy(), average="macro")
    row = {"model": model_name, "fraction": fraction, "seed": seed, "accuracy": accuracy, "macro_f1": macro_f1}
    _save_row(metrics_csv, row)

    # Save the model weights to disk if requested (default: only when training on 100% of data)
    if save_weights is None:
        save_weights = fraction == 1.0
    if save_weights:
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / f"{model_name}_frac{fraction:g}_seed{seed}.pth")

    seconds = time.perf_counter() - started
    print(f"TEST (best epoch {best_epoch}): accuracy {accuracy:.4f}, macro-F1 {macro_f1:.4f} "
          f"| {seconds / 60:.1f} min, {seconds / epochs:.1f} s per epoch")

    return {**row, "best_epoch": best_epoch, "val_accuracy": best_accuracy,
            "seconds_per_epoch": seconds / epochs, "history": history}


if __name__ == "__main__":
    # Quick smoke test: trains for 2 epochs on 5% of the data,
    # writes to a TEMPORARY file (the real metrics.csv is NOT touched).
    # Then runs again with 1 epoch to verify that re-runs replace rows instead of duplicating them.
    with tempfile.TemporaryDirectory() as folder:
        results_file = Path(folder) / "metrics.csv"

        train("scratch_cnn", fraction=0.05, seed=0, epochs=2, metrics_csv=results_file, save_weights=False)
        table = pd.read_csv(results_file)
        assert len(table) == 1 and 0 < table["accuracy"][0] <= 1, "one row with a sensible accuracy"

        train("scratch_cnn", fraction=0.05, seed=0, epochs=1, metrics_csv=results_file, save_weights=False)
        assert len(pd.read_csv(results_file)) == 1, "a rerun must replace its row, not add a duplicate"

    print("train: OK (one row written, a rerun replaced it)")
