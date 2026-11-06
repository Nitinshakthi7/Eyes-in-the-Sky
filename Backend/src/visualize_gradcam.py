"""visualize_gradcam.py - show Grad-CAM heatmaps for the first 4 wrong predictions.

For each of the first 4 misclassified test images, this script shows two panels:
  Left:  the original satellite image with its actual and predicted class labels
  Right: the same image with the Grad-CAM heatmap overlaid (red = model focused here)

This helps us understand WHY the model got it wrong:
  - Was it looking at the wrong part of the image?
  - Was the highlighted region actually misleading?

Output: results/figures/gradcam_examples.png
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models import build_transfer_model
from src.gradcam import grad_cam
from src.utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

# Run on CPU — only 4 images, so speed is not a concern here
device = torch.device("cpu")

# --- Load the trained MobileNetV2 model ---
# We use the full fine-tuned version trained on 100% of data with seed 0
model = build_transfer_model("mobilenet_v2", freeze=False)
checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- Image preprocessing ---
# Resize to 64x64, convert to float tensor, then normalise to ImageNet stats
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the misclassified images list (produced by evaluate.py)
df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# Pick the first 4 — just enough to show a variety of mistakes in the report
df = df.head(4).reset_index(drop=True)

# The 10 EuroSAT class names
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

# 4 rows (one per image), 2 columns (original and Grad-CAM overlay)
fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i, row in df.iterrows():

    # Load the original image
    image_path = DATA_DIR / row["path"]
    original = Image.open(image_path).convert("RGB")

    # Preprocess for the model: (1, 3, 64, 64) normalised tensor
    image_tensor = transform(original).unsqueeze(0).to(device)

    # Generate Grad-CAM heatmap using the last feature block of MobileNetV2
    # model.features[-1] is the last convolutional block — it captures the most abstract patterns
    heatmap, predicted_class = grad_cam(
        model,
        image_tensor,
        model.features[-1]
    )

    heatmap = heatmap.numpy()  # convert from tensor to numpy for matplotlib

    # --- Left panel: original image with actual vs predicted labels ---
    axes[i, 0].imshow(original)
    axes[i, 0].set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}"
    )
    axes[i, 0].axis("off")

    # --- Right panel: original image with Grad-CAM heatmap overlaid ---
    # We show the original image first, then place the heatmap on top at 50% opacity
    # alpha=0.5 makes it semi-transparent so you can still see the satellite image below
    axes[i, 1].imshow(original)
    axes[i, 1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[i, 1].set_title("Grad-CAM")
    axes[i, 1].axis("off")

plt.tight_layout()

# Save to results/figures/
output_path = RESULTS_DIR / "figures" / "gradcam_examples.png"
plt.savefig(output_path, dpi=120, bbox_inches="tight")
plt.show()

print("Grad-CAM figure saved:", output_path)
