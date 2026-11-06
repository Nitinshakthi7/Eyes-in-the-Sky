"""gradcam_confidence_drop.py - does hiding the highlighted region actually hurt the model?

Grad-CAM tells us WHICH pixels the model focused on, but it doesn't automatically prove
those pixels MATTER. This experiment tests that claim:

  1. Take 50 correctly classified test images
  2. Run Grad-CAM to find the "hot" (most important) region
  3. Black out the top 20% most important pixels
  4. Re-run the model and check: did confidence drop?

If Grad-CAM highlights meaningful regions, hiding those pixels should cause a bigger
confidence drop than hiding random pixels.

NOTE: This experiment has no control group — see gradcam_sanity_check.py for the
full experiment that also compares against hiding the COLDEST pixels.

Output: results/gradcam_confidence_drop.csv and results/figures/gradcam_confidence_drop.png

Run from the Backend folder: python -m src.gradcam_confidence_drop
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from pathlib import Path

from src.models import build_transfer_model
from src.gradcam import grad_cam
from src.utils import DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR

# Run on CPU (this script processes 50 images, which is fast enough)
device = torch.device("cpu")

# How many correctly classified test images to process
SAMPLE_SIZE = 50

# What percentage of pixels to black out (the most important ones)
MASK_PERCENT = 20

# Fix random number generators so the sample selection is reproducible
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# The 10 EuroSAT class names (same order as the model's output)
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

# --- Load the best trained model ---
model = build_transfer_model("mobilenet_v2", freeze=False)
checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- Image preprocessing: resize, convert to tensor, normalise ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Load the test images, shuffled with our fixed seed ---
# We shuffle so we don't always pick the same 50 images
df = pd.read_csv(DATA_DIR / "splits.csv")
df = df[df["split"] == "test"].sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

results = []

for _, row in df.iterrows():
    image_path = DATA_DIR / row["path"]
    original = Image.open(image_path).convert("RGB")
    image = transform(original).unsqueeze(0).to(device)

    # --- Step 1: Get the original prediction and confidence ---
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  # convert raw scores to probabilities (sum to 1)
        predicted = output.argmax(dim=1).item()

    actual = int(row["label"])

    # Skip images the model got wrong — we only test correctly classified ones
    if predicted != actual:
        continue

    # Confidence = how sure the model was about its correct prediction (0 to 1)
    original_confidence = probabilities[0, actual].item()

    # --- Step 2: Generate a Grad-CAM heatmap ---
    heatmap, _ = grad_cam(model, image, model.features[-1])
    heatmap = heatmap.to(device)

    # --- Step 3: Identify the top 20% most important pixels ---
    # torch.quantile finds the value where 80% of pixels fall below it
    # Any pixel above that threshold is in the "hot" top 20%
    threshold = torch.quantile(
        heatmap.flatten(), 1 - MASK_PERCENT / 100
    )
    mask = heatmap >= threshold  # True where pixels are "hot"

    # --- Step 4: Black out those pixels ---
    # We set them to 0.0, which after normalisation corresponds to the average image colour
    masked_image = image.clone()
    masked_image[0, :, mask] = 0.0  # zero out all 3 colour channels for the masked pixels

    # --- Step 5: Re-run the model on the masked image ---
    with torch.no_grad():
        masked_output = model(masked_image)
        masked_probabilities = torch.softmax(masked_output, dim=1)
        masked_confidence = masked_probabilities[0, actual].item()

    # How much did confidence fall after hiding the important region?
    confidence_drop = original_confidence - masked_confidence

    results.append({
        "path": row["path"],
        "class_name": class_names[actual],
        "original_confidence": original_confidence,
        "masked_confidence": masked_confidence,
        "confidence_drop": confidence_drop
    })

    # Stop once we have 50 correctly classified images
    if len(results) >= SAMPLE_SIZE:
        break

# --- Save the per-image results to a CSV ---
results_df = pd.DataFrame(results)
output_csv = RESULTS_DIR / "gradcam_confidence_drop.csv"
results_df.to_csv(output_csv, index=False)

if len(results_df) > 0:
    mean_original = results_df["original_confidence"].mean()
    mean_masked = results_df["masked_confidence"].mean()

    # --- Bar chart: original confidence vs. masked confidence ---
    # A big drop shows the highlighted region genuinely mattered to the model
    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Original", "Grad-CAM Masked"],
        [mean_original, mean_masked]
    )
    plt.ylabel("Average Confidence")
    plt.title("Confidence Drop After Grad-CAM Region Masking")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_fig = FIGURES_DIR / "gradcam_confidence_drop.png"
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print("Images tested:", len(results_df))
    print(f"Average original confidence: {mean_original:.4f}")
    print(f"Average masked confidence:  {mean_masked:.4f}")
    print(f"Average confidence drop:    {mean_original - mean_masked:.4f}")
    print("CSV saved:", output_csv)
    print("Graph saved:", output_fig)
else:
    print("No correctly classified images were processed.")
