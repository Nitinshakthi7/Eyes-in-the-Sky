"""Real examples (first three in the file) of the six most confused class pairs.

Run from the repo folder:  python -m src.visualize_confusion_examples
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import DATA_DIR, FIGURES_DIR, RESULTS_DIR

df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

pairs = [
    ("AnnualCrop", "PermanentCrop"),
    ("AnnualCrop", "Pasture"),
    ("HerbaceousVegetation", "Pasture"),
    ("HerbaceousVegetation", "PermanentCrop"),
    ("Highway", "River"),
    ("Pasture", "AnnualCrop"),
]

fig, axes = plt.subplots(6, 3, figsize=(12, 20))

for row, (actual, predicted) in enumerate(pairs):
    examples = df[
        (df["actual_class"] == actual) &
        (df["predicted_class"] == predicted)
    ].head(3)

    for col, (_, item) in enumerate(examples.iterrows()):
        image_path = DATA_DIR / item["path"]
        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"Actual: {actual}\nPredicted: {predicted}",
            fontsize=9
        )
        axes[row, col].axis("off")

    for col in range(len(examples), 3):
        axes[row, col].axis("off")

plt.suptitle(
    "Examples of Most Confused EuroSAT Class Pairs",
    fontsize=16
)
plt.tight_layout()

output = FIGURES_DIR / "confusion_pair_examples.png"
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved visualization: {output}")
