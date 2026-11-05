
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import DATA_DIR, RESULTS_DIR

# Load the incorrect predictions
df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# Select 12 incorrect images
df = df.head(12).reset_index(drop=True)

# Create a 3-row, 4-column grid
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, row in df.iterrows():

    # Find the image
    image_path = DATA_DIR / row["path"]

    # Open image
    img = Image.open(image_path).convert("RGB")

    # Display image
    axes[i].imshow(img)

    # Show actual and predicted classes
    axes[i].set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}",
        fontsize=9,
        pad=8
    )

    axes[i].axis("off")

# Hide unused plots
for i in range(len(df), len(axes)):
    axes[i].axis("off")

# Improve spacing
plt.subplots_adjust(
    hspace=0.35,
    wspace=0.15,
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
