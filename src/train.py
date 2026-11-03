"""train.py - the one shared training function.

Owner: Nitin
Part:  D (shared train() + data-efficiency study + repo)

Contains:
  train(model_name, fraction, seed)  - train one model with the team settings, add ONE row to results/metrics.csv
  build_model(model_name)            - the model and its learning rate for a model name
  SCRATCH, TRANSFER                  - the model names train() understands

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

BATCH_SIZE = 64
MAX_EPOCHS = 20

# Model names. The scratch CNN names are the ones Tejash used in his runs.
SCRATCH = {
    "scratch_cnn": {"use_batchnorm": True, "depth": 3},                # the baseline
    "scratch_cnn_no_batchnorm": {"use_batchnorm": False, "depth": 3},  # ablation 1
    "scratch_cnn_depth2": {"use_batchnorm": True, "depth": 2},         # ablation 2
}
# Transfer models (Rachana, part C): name -> (backbone, freeze the backbone?)
TRANSFER = {
    "resnet18_frozen": ("resnet18", True),
    "resnet18_full": ("resnet18", False),
    "mobilenet_v2_frozen": ("mobilenet_v2", True),
    "mobilenet_v2_full": ("mobilenet_v2", False),
}


def build_model(model_name):
    """Return (model, learning rate) for one of our model names."""
    if model_name in SCRATCH:
        return models.build_scratch_cnn(**SCRATCH[model_name]), 1e-3
    if model_name in TRANSFER:
        if not hasattr(models, "build_transfer_model"):
            raise NotImplementedError("build_transfer_model is not in src/models.py yet (Rachana, part C)")
        backbone, freeze = TRANSFER[model_name]
        # Team default: 0.001 when only the new last layer learns, 0.0001 when everything is fine-tuned.
        return models.build_transfer_model(backbone, freeze), (1e-3 if freeze else 1e-4)
    raise ValueError(f"unknown model name {model_name!r}; use one of {sorted([*SCRATCH, *TRANSFER])}")


# ---------- fast data: all images in memory ----------
_cache = {}  # filled by the first train() call and reused by every later call in the same Python session


def _load_all_images(device):
    """Read all 27,000 images once. Returns (images, labels, row_of_path) with everything on the device."""
    key = str(device)
    if key not in _cache:
        df = pd.read_csv(SPLITS_CSV)
        print(f"Reading {len(df)} images into memory (one time, about 20 seconds)...")
        pictures = [np.array(Image.open(DATA_DIR / path).convert("RGB")) for path in df["path"]]
        images = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2).contiguous()  # (N, 3, 64, 64) bytes
        labels = torch.tensor(df["label"].values)
        row_of_path = {path: row for row, path in enumerate(df["path"])}
        _cache[key] = (images.to(device), labels.to(device), row_of_path)
    return _cache[key]


def _rows(split, fraction, seed, row_of_path, device):
    """Row numbers of one split. For train, a stratified share of it, chosen by get_loader's own rule."""
    # get_loader only reads splits.csv here (no images), so we reuse its exact subset rule.
    frame = get_loader(split, fraction=fraction if split == "train" else 1.0, seed=seed).dataset.df
    return torch.tensor([row_of_path[path] for path in frame["path"]], device=device)


def _prepare(batch_bytes, augment):
    """Bytes (0 to 255) to normalised floats. With augment: random 90-degree turns and flips, per image."""
    x = batch_bytes.float() / 255.0
    if augment:
        n = x.shape[0]
        turns = torch.randint(0, 4, (n,), device=x.device)
        for k in (1, 2, 3):
            chosen = turns == k
            if chosen.any():
                x[chosen] = torch.rot90(x[chosen], k, dims=(2, 3))
        flip_sideways = torch.rand(n, device=x.device) < 0.5
        flip_upside_down = torch.rand(n, device=x.device) < 0.5
        x[flip_sideways] = x[flip_sideways].flip(3)
        x[flip_upside_down] = x[flip_upside_down].flip(2)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _score(model, rows, images, labels, loss_function):
    """Average loss, accuracy and predictions for some rows, without augmentation."""
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            scores = model(_prepare(images[batch], augment=False))
            total_loss += loss_function(scores, labels[batch]).item() * len(batch)
            predictions.append(scores.argmax(dim=1))
    predictions = torch.cat(predictions)
    accuracy = (predictions == labels[rows]).float().mean().item()
    return total_loss / len(rows), accuracy, predictions


def _save_row(path, row):
    """Add one row to the results file. A rerun of the same (model, fraction, seed) replaces its old row."""
    columns = ["model", "fraction", "seed", "accuracy", "macro_f1"]
    path = Path(path)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=columns)
    if old.empty:
        new = pd.DataFrame([row], columns=columns)
    else:
        same_run = ((old["model"] == row["model"]) & np.isclose(old["fraction"].astype(float), row["fraction"])
                    & (old["seed"] == row["seed"]))
        new = pd.concat([old.loc[~same_run], pd.DataFrame([row], columns=columns)], ignore_index=True)
    new.to_csv(path, index=False, lineterminator="\n")


def train(model_name, fraction, seed, epochs=MAX_EPOCHS, metrics_csv=METRICS_CSV, save_weights=None):
    """Train one model with the team settings and add ONE row to results/metrics.csv.

    model_name:   one of SCRATCH or TRANSFER above
    fraction:     share of the TRAIN split to use, from 0.05 to 1.0 (stratified). Validation and test never shrink.
    seed:         makes the run repeatable (starting weights, which images, their order, augmentation)
    save_weights: keep the best-epoch weights in models/ (not in git). Default: only when fraction is 1.0.

    The best epoch is chosen on the VALIDATION set. The test set is scored once, after training.
    Returns a dict with the result and the training history.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be above 0 and at most 1")
    device = get_device()
    set_seed(seed)

    images, labels, row_of_path = _load_all_images(device)
    train_rows = _rows("train", fraction, seed, row_of_path, device)
    val_rows = _rows("val", 1.0, seed, row_of_path, device)
    test_rows = _rows("test", 1.0, seed, row_of_path, device)

    model, learning_rate = build_model(model_name)
    model = model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # Only the layers that are allowed to learn go to the optimiser (matters for the frozen models).
    # foreach=False is the same setting Tejash used in his runs.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, foreach=False)
    shuffler = torch.Generator().manual_seed(seed)  # decides the order of the images, so a run repeats

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy, best_epoch, best_weights = -1.0, 0, None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0

        order = train_rows[torch.randperm(len(train_rows), generator=shuffler).to(device)]
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            if len(batch) < 2:  # BatchNorm cannot work with a single image
                continue
            x = _prepare(images[batch], augment=True)
            y = labels[batch]
            optimizer.zero_grad()
            scores = model(x)
            loss = loss_function(scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(dim=1) == y).sum()
            seen += len(batch)

        val_loss, val_accuracy, _ = _score(model, val_rows, images, labels, loss_function)
        history["train_loss"].append(total_loss.item() / seen)
        history["train_accuracy"].append(correct.item() / seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"{model_name} | fraction {fraction:g} | seed {seed} | epoch {epoch:02d}/{epochs} | "
              f"train loss {history['train_loss'][-1]:.4f}, acc {history['train_accuracy'][-1]:.3f} | "
              f"val loss {val_loss:.4f}, acc {val_accuracy:.3f} | {time.perf_counter() - epoch_started:.1f} s")

        if val_accuracy > best_accuracy:  # keep the best epoch, chosen on validation
            best_accuracy, best_epoch = val_accuracy, epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    model.load_state_dict(best_weights)

    # The test set is used here, once, and nothing is decided from it.
    _, accuracy, predictions = _score(model, test_rows, images, labels, loss_function)
    macro_f1 = f1_score(labels[test_rows].cpu().numpy(), predictions.cpu().numpy(), average="macro")
    row = {"model": model_name, "fraction": fraction, "seed": seed, "accuracy": accuracy, "macro_f1": macro_f1}
    _save_row(metrics_csv, row)

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
    # Tiny check: 2 epochs on 5% of the data, written to a temporary file (the real metrics.csv is NOT touched).
    with tempfile.TemporaryDirectory() as folder:
        results_file = Path(folder) / "metrics.csv"
        train("scratch_cnn", fraction=0.05, seed=0, epochs=2, metrics_csv=results_file, save_weights=False)
        table = pd.read_csv(results_file)
        assert len(table) == 1 and 0 < table["accuracy"][0] <= 1, "one row with a sensible accuracy"
        train("scratch_cnn", fraction=0.05, seed=0, epochs=1, metrics_csv=results_file, save_weights=False)
        assert len(pd.read_csv(results_file)) == 1, "a rerun must replace its row, not add a duplicate"
    print("train: OK (one row written, a rerun replaced it)")
