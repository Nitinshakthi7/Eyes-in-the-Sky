import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

df = pd.read_csv("results/misclassified_images.csv")

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
        image_path = Path("data") / item["path"]
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

output = Path("results/figures/confusion_pair_examples.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved visualization: {output}")
