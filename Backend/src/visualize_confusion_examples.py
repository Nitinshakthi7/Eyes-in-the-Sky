"""visualize_confusion_examples.py - show real images for the most commonly confused class pairs.

After looking at the confusion matrix, we identified the 6 class pairs the model mixes up most.
This script pulls 3 real example images per pair from the misclassified_images.csv list
and displays them in a grid so you can see WHY those pairs are hard to tell apart.

For example, AnnualCrop vs PermanentCrop can look almost identical from above —
both have green, regularly-spaced patterns.

Output: results/figures/confusion_pair_examples.png

Run from the Backend folder: python -m src.visualize_confusion_examples
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import DATA_DIR, FIGURES_DIR, RESULTS_DIR

# Load all misclassified images (produced by evaluate.py)
df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# The 6 class pairs we want to highlight — these were picked from the confusion matrix
# as the ones with the most off-diagonal errors
pairs = [
    ("AnnualCrop", "PermanentCrop"),        # both look like rows of crops from above
    ("AnnualCrop", "Pasture"),              # both are open flat land, different colours
    ("HerbaceousVegetation", "Pasture"),    # both are grassy; fine texture vs coarser
    ("HerbaceousVegetation", "PermanentCrop"),
    ("Highway", "River"),                   # both are long, thin, grey/blue stripes
    ("Pasture", "AnnualCrop"),             # symmetric confusion: sometimes the model goes the other way
]

# 6 pairs × 3 examples each = 18 images total; laid out as 6 rows × 3 columns
fig, axes = plt.subplots(6, 3, figsize=(12, 20))

for row, (actual, predicted) in enumerate(pairs):
    # Find all misclassified images where actual class = `actual` and model said `predicted`
    examples = df[
        (df["actual_class"] == actual) &
        (df["predicted_class"] == predicted)
    ].head(3)  # take at most 3 examples

    for col, (_, item) in enumerate(examples.iterrows()):
        image_path = DATA_DIR / item["path"]
        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"Actual: {actual}\nPredicted: {predicted}",
            fontsize=9
        )
        axes[row, col].axis("off")

    # If this pair has fewer than 3 examples, hide the unused slots
    for col in range(len(examples), 3):
        axes[row, col].axis("off")

plt.suptitle(
    "Examples of Most Confused EuroSAT Class Pairs",
    fontsize=16
)
plt.tight_layout()

# Save the figure to results/figures/
output = FIGURES_DIR / "confusion_pair_examples.png"
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved visualization: {output}")
