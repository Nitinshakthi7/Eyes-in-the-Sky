"""evaluate.py - how good is the model, and where does it fail?

Owner: Rohan
Part: E (Evaluation + Grad-CAM + t-SNE)

This file runs the trained model on the test set and produces:
  - Overall accuracy and macro-F1 score
  - A per-class report (precision, recall, F1 for each land-use type)
  - A confusion matrix (which classes get mixed up with which?)
  - A CSV of all misclassified images (for Grad-CAM and error analysis)

Run it from the Backend folder:
  python -m src.evaluate --model mobilenet_v2_full --fraction 1.0 --seed 0
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
    """Run the trained model on the test set and save all evaluation outputs.

    Steps:
    1. Load the saved model weights from the models/ folder
    2. Run all 4,050 test images through the model (no gradients, no augmentation)
    3. Compare predictions to true labels
    4. Compute accuracy, macro-F1, confusion matrix, and per-class report
    5. Save the misclassified images to a CSV (used by gradcam_all_classes.py)
    6. Save the confusion matrix as a CSV and as a heatmap PNG

    Parameters:
      model_name:  which model to evaluate (e.g. "mobilenet_v2_full", "scratch_cnn")
      fraction:    which checkpoint to load (1.0 = trained on all training data)
      seed:        which seed checkpoint to load
      batch_size:  how many images to process at once
    """
    device = get_device()

    # Rebuild the model architecture (same structure as when it was trained)
    model, _ = build_model(model_name)

    # Build the expected file name for the saved weights
    checkpoint_path = MODELS_DIR / (
        f"{model_name}_frac{fraction:g}_seed{seed}.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the saved weights into the model
    # map_location=device: load directly onto GPU if available
    # weights_only=True: safe loading (doesn't execute arbitrary code from the file)
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()  # switch to evaluation mode (turns off dropout, uses BN running stats)

    print("Model:", model_name)
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)

    # Build the test DataLoader — no augmentation, no shuffling, all 4050 test images
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

    # Run the model on every batch in the test set
    with torch.no_grad():  # no gradient tracking needed during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)              # get class scores for the batch
            predictions = outputs.argmax(dim=1)  # pick the class with the highest score

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    y_true = np.array(actual_labels)
    y_pred = np.array(predicted_labels)

    # --- Find and save all misclassified images ---
    # We need this CSV for Grad-CAM analysis (visualize_gradcam.py, gradcam_all_classes.py)
    test_df = test_loader.dataset.df.reset_index(drop=True)

    # Get the list of class names from the dataset (e.g. "Forest", "Highway", ...)
    class_names = EuroSAT(
        root=str(DATA_DIR),
        download=False,
    ).classes

    # Find rows where prediction doesn't match the true label
    mistakes = test_df.iloc[np.where(y_true != y_pred)[0]].copy()

    # Add extra columns so we know what it actually was vs. what the model said
    mistakes["actual_class"] = [
        class_names[label] for label in y_true[y_true != y_pred]
    ]
    mistakes["predicted_class"] = [
        class_names[label] for label in y_pred[y_true != y_pred]
    ]
    mistakes["actual_label"] = y_true[y_true != y_pred]
    mistakes["predicted_label"] = y_pred[y_true != y_pred]

    # Save misclassified images to CSV
    mistakes_path = RESULTS_DIR / "misclassified_images.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mistakes.to_csv(mistakes_path, index=False)

    print("\nMisclassified images saved:", mistakes_path)
    print("Total misclassified images:", len(mistakes))

    # Reload class names (clean reference for the metrics section)
    class_names = EuroSAT(
        root=str(DATA_DIR),
        download=False,
    ).classes

    # --- Compute evaluation metrics ---
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-F1: average F1 across all 10 classes, treating each class equally
    # (unlike weighted F1, macro F1 doesn't favour larger classes)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # Confusion matrix: a 10x10 grid where entry [i][j] = how many class-i images were predicted as class-j
    # The diagonal is correct predictions; off-diagonal are mistakes.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    # Per-class report: precision, recall, and F1 for every land-use type
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,  # return a dict so we can save it to CSV
        zero_division=0,
    )

    # Make sure output folders exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build a file name prefix for all outputs from this run
    name = f"{model_name}_frac{fraction:g}_seed{seed}"

    # --- Save confusion matrix as CSV ---
    cm_path = RESULTS_DIR / f"{name}_confusion_matrix.csv"
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(cm_path)

    # --- Save per-class classification report as CSV ---
    report_path = RESULTS_DIR / f"{name}_classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    # --- Plot the confusion matrix as a coloured heatmap ---
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(cm, cmap="Blues")  # darker = more images predicted in that cell
    fig.colorbar(image, ax=ax)           # add colour scale legend

    # Label both axes with class names
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion Matrix - {model_name}")

    # Write the count number inside each cell of the matrix
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()

    plot_path = FIGURES_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    # --- Print a summary to the terminal ---
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
    # Allow running from the command line with optional arguments
    # Example: python -m src.evaluate --model scratch_cnn --fraction 1.0 --seed 0
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
