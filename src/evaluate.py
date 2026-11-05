
"""evaluate.py - how good is the model, and where does it fail?

Owner: Rohan
Part: E (Evaluation + Grad-CAM + t-SNE)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision.datasets import EuroSAT

from src.dataset import get_loader
from src.train import build_model
from src.utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, get_device


def evaluate_model(model_name, fraction=1.0, seed=0, batch_size=64):

    device = get_device()

    # Build model architecture
    model, _ = build_model(model_name)

    # Find checkpoint
    checkpoint_path = MODELS_DIR / (
        f"{model_name}_frac{fraction:g}_seed{seed}.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load trained weights
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("Model:", model_name)
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)

    # Load test dataset
    test_loader = get_loader(
        "test",
        fraction=1.0,
        seed=0,
        augment=False,
        batch_size=batch_size,
        num_workers=0,
    )

    actual_labels = []
    predicted_labels = []

    # Predict all test images
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    y_true = np.array(actual_labels)
    y_pred = np.array(predicted_labels)

    # Save the paths of incorrectly predicted test images
    test_df = test_loader.dataset.df.reset_index(drop=True)

    class_names = EuroSAT(
        root=str(DATA_DIR),
        download=False,
    ).classes

    mistakes = test_df.iloc[np.where(y_true != y_pred)[0]].copy()

    mistakes["actual_class"] = [
        class_names[label] for label in y_true[y_true != y_pred]
    ]

    mistakes["predicted_class"] = [
        class_names[label] for label in y_pred[y_true != y_pred]
    ]

    mistakes["actual_label"] = y_true[y_true != y_pred]
    mistakes["predicted_label"] = y_pred[y_true != y_pred]

    mistakes_path = RESULTS_DIR / "misclassified_images.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mistakes.to_csv(mistakes_path, index=False)

    print("\nMisclassified images saved:", mistakes_path)
    print("Total misclassified images:", len(mistakes))

    # EuroSAT class names
    class_names = EuroSAT(
        root=str(DATA_DIR),
        download=False,
    ).classes

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Create output folders
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    name = f"{model_name}_frac{fraction:g}_seed{seed}"

    # Save confusion matrix
    cm_path = RESULTS_DIR / f"{name}_confusion_matrix.csv"

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(cm_path)

    # Save classification report
    report_path = RESULTS_DIR / f"{name}_classification_report.csv"

    pd.DataFrame(report).transpose().to_csv(report_path)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion Matrix - {model_name}")

    # Write numbers inside matrix cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()

    plot_path = FIGURES_DIR / f"{name}_confusion_matrix.png"

    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    # Display results
    print("\n========== EVALUATION RESULTS ==========")
    print("Test images:", len(y_true))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")

    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix CSV:", cm_path)
    print("Classification report CSV:", report_path)
    print("Confusion matrix plot:", plot_path)

    return y_true, y_pred, cm, report


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="mobilenet_v2_full")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        fraction=args.fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
