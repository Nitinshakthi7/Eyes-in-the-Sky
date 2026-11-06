"""gradcam_sanity_check.py - is Grad-CAM actually telling the truth on our model?

Part E control experiment (added by Nitin with Claude).

The confidence-drop experiment (gradcam_confidence_drop.py) only hides the "hot" Grad-CAM region.
But a drop in confidence doesn't prove Grad-CAM is correct — hiding ANY region might hurt the model.
To be sure the highlighted region is SPECIAL, we need a control group.

This script runs two checks on 200 test images (first 20 from each of the 10 classes):

  Check 1 - Spatial spread:
    Which quarter (top-left / top-right / bottom-left / bottom-right) of the 64x64 image
    is the hottest in each Grad-CAM map? If the heat is randomly distributed, we'd expect
    25% in each quarter. If Grad-CAM is meaningful, the distribution should vary by class.

  Check 2 - Three masking experiments (hides 25% of the pixels each time):
    - Hottest 25%:  the region Grad-CAM says matters most
    - Coldest 25%:  the region Grad-CAM says matters LEAST
    - Random 32x32 square: a random patch anywhere in the image
    If Grad-CAM is meaningful, hiding the hottest pixels should hurt MUCH MORE
    than hiding the coldest or a random patch.

Output: results/gradcam_sanity_check.csv (one row per image with all the drop values)

Run from the Backend folder: python -m src.gradcam_sanity_check  (CPU, about 1 minute)
"""

import numpy as np
import pandas as pd
import torch

from src.dataset import EuroSATDataset
from src.gradcam import grad_cam
from src.train import build_model
from src.utils import MODELS_DIR, RESULTS_DIR

# How many test images to use per class (10 classes × 20 = 200 images total)
IMAGES_PER_CLASS = 20

# How many pixels to hide in each masking experiment
# 64 × 64 = 4096 total pixels; 25% = 1024
HIDE = 1024

# Human-readable names for the four image quadrants
QUADRANTS = {(0, 0): "top-left", (0, 1): "top-right", (1, 0): "bottom-left", (1, 1): "bottom-right"}


def painted(batch, flat_ids):
    """Return a copy of the image with specific pixels blacked out.

    flat_ids: a tensor of flat pixel indices to zero out (into a 64*64 = 4096-element array)
    Setting to 0.0 after normalisation corresponds to the average training image colour,
    so we're replacing those pixels with something neutral — not all-black or all-white.
    """
    out = batch.clone().reshape(1, 3, -1)   # reshape to (1, 3, 4096) so we can index by flat pixel index
    out[:, :, flat_ids] = 0.0               # zero all 3 colour channels for the selected pixels
    return out.reshape(1, 3, 64, 64)        # reshape back to image dimensions


def main():
    # --- Load the trained model ---
    model, _ = build_model("mobilenet_v2_full")
    state = torch.load(MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load the test split (uses splits.csv, no images loaded yet)
    data = EuroSATDataset("test")

    # Fixed seed for the random square generator — makes the experiment reproducible
    generator = torch.Generator().manual_seed(0)

    rows = []

    # Loop over all 10 classes and pick the first 20 test images from each
    for label in range(10):
        for index in data.df.index[data.df["label"] == label][:IMAGES_PER_CLASS]:

            # Fetch the preprocessed image tensor: shape (3, 64, 64), already normalised
            batch = data[int(index)][0].unsqueeze(0)  # add batch dim → (1, 3, 64, 64)

            # Generate Grad-CAM heatmap for this image
            heat, predicted = grad_cam(model, batch, model.features[-1])
            h = heat.numpy()  # convert to numpy for the quadrant analysis

            # --- Check 1: Which quadrant is hottest? ---
            # Divide the 64x64 heatmap into four 32x32 quarters and average each
            quarters = np.array([
                [h[:32, :32].mean(), h[:32, 32:].mean()],
                [h[32:, :32].mean(), h[32:, 32:].mean()]
            ])

            # --- Check 2: Set up the three pixel masks ---
            flat = heat.flatten()

            # Random 32x32 square: pick a random top-left corner, then list all pixels in the square
            top, left = torch.randint(0, 33, (2,), generator=generator).tolist()
            square = ((torch.arange(32)[:, None] + top) * 64 + (torch.arange(32)[None, :] + left)).flatten()

            patches = {
                "hottest":       flat.topk(HIDE).indices,       # the HIDE pixels with the highest heat
                "coldest":       (-flat).topk(HIDE).indices,    # the HIDE pixels with the LOWEST heat
                "random_square": square                          # a random 32x32 block
            }

            # --- Measure confidence before masking ---
            with torch.no_grad():
                before = torch.softmax(model(batch), dim=1)[0, predicted].item()

            # Build the result row for this image
            row = {
                "path": data.df.loc[index, "path"],
                "class_name": data.df.loc[index, "class_name"],
                # Which quadrant had the most heat?
                "hottest_quadrant": QUADRANTS[np.unravel_index(quarters.argmax(), (2, 2))]
            }

            # For each mask type, measure how much confidence drops and whether prediction changes
            for name, ids in patches.items():
                probs = torch.softmax(model(painted(batch, ids)), dim=1)[0]
                row[f"drop_{name}"] = before - probs[predicted].item()      # confidence drop (positive = hurts)
                row[f"changed_{name}"] = int(probs.argmax().item() != predicted)  # 1 if prediction changed

            rows.append(row)

    # --- Save results ---
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "gradcam_sanity_check.csv", index=False)

    # --- Print a summary ---
    n = len(table)
    print(f"Images tested: {n}")

    print("Check 1 - hottest quadrant of the Grad-CAM map:")
    for name in QUADRANTS.values():
        count = int((table["hottest_quadrant"] == name).sum())
        print(f"  {name:13s} {count:3d}  ({count / n:.0%})")

    print("Check 2 - hide 25% of the pixels:")
    for name in ("hottest", "coldest", "random_square"):
        print(f"  {name:14s} mean confidence drop {table[f'drop_{name}'].mean():.3f} | prediction changed for {int(table[f'changed_{name}'].sum())} of {n}")

    # Key result: does hiding hottest hurt MORE than hiding coldest?
    hot, cold = table["drop_hottest"], table["drop_coldest"]
    print(f"  hottest hurt MORE than coldest for {int((hot > cold).sum())} images, LESS for {int((hot < cold).sum())}")


if __name__ == "__main__":
    main()
