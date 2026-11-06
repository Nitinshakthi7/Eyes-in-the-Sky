"""gradcam_all_classes.py - Grad-CAM heatmaps for every land-use class.

For each of the 10 EuroSAT classes, this script picks:
  - One WRONG prediction (from the misclassified_images.csv list)
  - One CORRECT prediction (first correctly classified image in the test set)

For each image it shows 3 columns:
  1. The original satellite image
  2. The raw Grad-CAM heatmap (red = model focused here)
  3. The original image with the heatmap overlaid (so you can see what the model looked at)

Output: results/figures/gradcam_all_classes.png

Run from the Backend folder: python -m src.gradcam_all_classes
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.models import build_transfer_model
from src.gradcam import grad_cam
from src.utils import DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SPLITS_CSV

# Use CPU for this script — Grad-CAM on a few images is fast enough without a GPU
device = torch.device("cpu")

# The 10 EuroSAT land-use classes, in the same order as the model's output neurons
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

# --- Load the trained MobileNetV2 model ---
# We always use the full fine-tuned version trained on 100% of the data with seed 0
# because that's our best model (highest accuracy)
model = build_transfer_model("mobilenet_v2", freeze=False)
checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- Image preprocessing ---
# Resize to 64x64 (what the model was trained on), convert to tensor, normalise
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Load the split and misclassification tables ---
splits = pd.read_csv(SPLITS_CSV)
test_df = splits[splits["split"] == "test"].reset_index(drop=True)
wrong_df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# We'll collect (class_name, PIL image, is_correct) tuples here
samples = []

# --- Step 1: Collect one wrong example per class ---
# For each class, grab the first image the model got wrong
for class_name in class_names:
    mistakes = wrong_df[wrong_df["actual_class"] == class_name]
    if len(mistakes) > 0:
        row = mistakes.iloc[0]
        image_path = DATA_DIR / row["path"]
        samples.append((
            class_name,
            Image.open(image_path).convert("RGB"),
            False  # is_correct = False → this is a wrong prediction
        ))

# --- Step 2: Find one correct example per class ---
# Walk through the test set in order and pick the first image the model gets right per class
correct_found = set()

for _, row in test_df.iterrows():
    class_name = row["class_name"]

    # Skip if we already have a correct example for this class
    if class_name in correct_found:
        continue

    image_path = DATA_DIR / row["path"]
    image = Image.open(image_path).convert("RGB")

    # Run the model to check if it predicts this image correctly
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension: (1, 3, 64, 64)

    with torch.no_grad():
        prediction = model(input_tensor).argmax(1).item()

    if prediction == row["label"]:
        samples.append((class_name, image, True))  # is_correct = True
        correct_found.add(class_name)

    # Stop once we have a correct example for all 10 classes
    if len(correct_found) == 10:
        break

# --- Build the figure ---
# Each sample gets one row; each row has 3 columns (original / heatmap / overlay)
fig, axes = plt.subplots(
    len(samples), 3,
    figsize=(12, len(samples) * 2.8)
)

# If there's only one sample, axes is 1D — expand it so the loop always gets 2D indexing
if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, (class_name, image, is_correct) in enumerate(samples):
    # Preprocess the image for Grad-CAM
    input_tensor = transform(image).unsqueeze(0)

    # Generate Grad-CAM: use the last feature block (most abstract features)
    heatmap, predicted_class = grad_cam(
        model,
        input_tensor,
        layer=model.features[-1]
    )

    # Resize image to 64x64 for consistent display
    image_np = np.array(image.resize((64, 64)))

    # Column 1: original image with correct/wrong label
    axes[i, 0].imshow(image_np)
    axes[i, 0].set_title(
        f"{class_name} - {'Correct' if is_correct else 'Wrong'}"
    )
    axes[i, 0].axis("off")

    # Column 2: raw Grad-CAM heatmap (red = high importance, blue = low)
    axes[i, 1].imshow(heatmap, cmap="jet")
    axes[i, 1].set_title("Grad-CAM")
    axes[i, 1].axis("off")

    # Column 3: original image with heatmap overlaid at 40% opacity
    axes[i, 2].imshow(image_np)
    axes[i, 2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[i, 2].set_title(f"Predicted: {class_names[predicted_class]}")
    axes[i, 2].axis("off")

plt.tight_layout()

# Save the figure to the results/figures folder
output = FIGURES_DIR / "gradcam_all_classes.png"
plt.savefig(output, dpi=150)
plt.close()

print("Grad-CAM saved:", output)
print("Images included:", len(samples))
print("Correct examples:", len(correct_found))
print("Misclassified examples:", len(samples) - len(correct_found))
