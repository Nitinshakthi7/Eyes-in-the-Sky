
"""Memory-efficient Part E scratch CNN seed-4 experiments."""

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
    device = get_device()
    set_seed(seed)

    train_loader = get_loader(
        "train", fraction=fraction, seed=seed,
        augment=True, batch_size=BATCH_SIZE
    )
    val_loader = get_loader("val", batch_size=BATCH_SIZE)
    test_loader = get_loader("test", batch_size=BATCH_SIZE)

    model = build_scratch_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_accuracy = -1
    best_weights = None
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            scores = model(x)
            loss = loss_fn(scores, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (scores.argmax(1) == y).sum().item()
            seen += len(y)

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

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_weights)
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        for x, y in test_loader:
            scores = model(x.to(device))
            predicted.extend(scores.argmax(1).cpu().tolist())
            actual.extend(y.tolist())

    accuracy = np.mean(np.array(actual) == np.array(predicted))
    macro_f1 = f1_score(actual, predicted, average="macro")

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
    train_part_e(fraction=0.25, seed=4)
