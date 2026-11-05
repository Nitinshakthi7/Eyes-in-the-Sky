
"""Grad-CAM confidence-drop experiment."""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from pathlib import Path

from models import build_transfer_model
from gradcam import grad_cam
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

device = torch.device("cpu")
SAMPLE_SIZE = 50
MASK_PERCENT = 20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

model = build_transfer_model("mobilenet_v2", freeze=False)
checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

df = pd.read_csv(DATA_DIR / "splits.csv")
df = df[df["split"] == "test"].sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

results = []

for _, row in df.iterrows():
    image_path = DATA_DIR / row["path"]
    original = Image.open(image_path).convert("RGB")
    image = transform(original).unsqueeze(0).to(device)

    # Original prediction and confidence
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted = output.argmax(dim=1).item()

    actual = int(row["label"])

    # Only use correctly classified images
    if predicted != actual:
        continue

    original_confidence = probabilities[0, actual].item()

    # Generate Grad-CAM
    heatmap, _ = grad_cam(model, image, model.features[-1])
    heatmap = heatmap.to(device)

    # Mask top 20% most important pixels
    threshold = torch.quantile(
        heatmap.flatten(), 1 - MASK_PERCENT / 100
    )
    mask = heatmap >= threshold

    masked_image = image.clone()
    masked_image[0, :, mask] = 0.0

    # Confidence after masking
    with torch.no_grad():
        masked_output = model(masked_image)
        masked_probabilities = torch.softmax(masked_output, dim=1)
        masked_confidence = masked_probabilities[0, actual].item()

    confidence_drop = original_confidence - masked_confidence

    results.append({
        "path": row["path"],
        "class_name": class_names[actual],
        "original_confidence": original_confidence,
        "masked_confidence": masked_confidence,
        "confidence_drop": confidence_drop
    })

    if len(results) >= SAMPLE_SIZE:
        break

results_df = pd.DataFrame(results)

output_csv = RESULTS_DIR / "gradcam_confidence_drop.csv"
results_df.to_csv(output_csv, index=False)

if len(results_df) > 0:
    mean_original = results_df["original_confidence"].mean()
    mean_masked = results_df["masked_confidence"].mean()

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Original", "Grad-CAM Masked"],
        [mean_original, mean_masked]
    )
    plt.ylabel("Average Confidence")
    plt.title("Confidence Drop After Grad-CAM Region Masking")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_fig = RESULTS_DIR / "figures" / "gradcam_confidence_drop.png"
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
