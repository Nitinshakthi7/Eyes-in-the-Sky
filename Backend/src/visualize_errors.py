"""visualize_errors.py - show a grid of images the model got wrong.

This script loads the misclassified_images.csv (produced by evaluate.py) and
displays the first 12 wrong predictions as a 3×4 image grid.

Each image is titled with:
  - The actual class (what it really was)
  - The predicted class (what the model thought it was)

This is a quick visual sanity check: do the mistakes look reasonable?
(e.g. the model confusing AnnualCrop for Pasture is understandable — they look similar)

Output: results/misclassified_images_grid.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import DATA_DIR, RESULTS_DIR

# Load the list of all images the model got wrong (produced by src/evaluate.py)
df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# Just take the first 12 to keep the grid a manageable size
df = df.head(12).reset_index(drop=True)

# Create a 3-row × 4-column grid of image slots
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()  # flatten from 2D array to 1D list so we can index with a single number

for i, row in df.iterrows():

    # Build the full path to this image on disk
    image_path = DATA_DIR / row["path"]

    # Open the image and ensure it's in RGB mode (some JPEGs can be greyscale or CMYK)
    img = Image.open(image_path).convert("RGB")

    # Display the image in grid slot i
    axes[i].imshow(img)

    # Title shows what it actually was and what the model predicted
    axes[i].set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}",
        fontsize=9,
        pad=8
    )

    # Remove axis ticks — cleaner look for image grids
    axes[i].axis("off")

# If we have fewer than 12 images, hide the leftover empty slots
for i in range(len(df), len(axes)):
    axes[i].axis("off")

# Adjust spacing so titles don't overlap the images above them
plt.subplots_adjust(
    hspace=0.35,  # vertical gap between rows
    wspace=0.15,  # horizontal gap between columns
    top=0.95,
    bottom=0.03
)

# Save the figure
output_path = RESULTS_DIR / "misclassified_images_grid.png"
plt.savefig(
    output_path,
    dpi=120,
    bbox_inches="tight"
)

plt.show()

print("Image grid saved:", output_path)
print("Images displayed:", len(df))
