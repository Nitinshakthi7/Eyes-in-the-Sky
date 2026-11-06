"""visualize_tsne.py - t-SNE plot of MobileNetV2's internal feature representations.

t-SNE (t-distributed Stochastic Neighbour Embedding) is a technique that takes
high-dimensional data and squashes it into 2D so we can plot it.

What we're plotting here:
  - For 1000 test images, we run them through MobileNetV2 (up to the last feature layer)
    and collect the internal features (a vector of 1280 numbers per image).
  - t-SNE then positions each image as a dot in 2D so that images with similar
    internal features end up close together.
  - We colour each dot by its true land-use class.

A good model should produce tight, well-separated clusters — images the model has
learned to represent similarly (same class) should cluster together.

Output: results/figures/mobilenet_v2_tsne.png

Run from the Backend folder: python -m src.visualize_tsne  (takes a few minutes on CPU)
"""

import os

# Limit CPU threads used by OpenBLAS/OpenMP/MKL to avoid slowdowns on some machines
# (t-SNE uses numpy internally which uses these)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE

from src.models import build_transfer_model
from src.utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR


# Step 1: Use CPU (t-SNE runs on numpy anyway; GPU doesn't help here)
device = torch.device("cpu")


# Step 2: Load the trained MobileNetV2 model
model = build_transfer_model("mobilenet_v2", freeze=False)

checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"

model.load_state_dict(
    torch.load(checkpoint, map_location=device, weights_only=True)
)

model.to(device)
model.eval()  # evaluation mode: no dropout, use BN running stats


# Step 3: Set up image preprocessing
# (same as during training — resize to 64x64, convert to tensor, normalise)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Step 4: Read the test image paths from splits.csv using Python's built-in CSV reader
# (not pandas, to show it works without any extra dependencies)
split_file = DATA_DIR / "splits.csv"

test_rows = []

with open(split_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["split"] == "test":
            test_rows.append(row)

print("Total test images:", len(test_rows))


# Randomly sample 1000 test images with a fixed seed so the plot is reproducible
# (processing all 4050 would take longer; 1000 is enough to see the clusters)
random.Random(42).shuffle(test_rows)
test_rows = test_rows[:1000]


# Step 5: Extract the internal feature vectors from MobileNetV2
# We pass each image through model.features (all conv blocks) but NOT through the classifier.
# This gives us the learned visual representation — not the final prediction.
features_list = []
labels_list = []

print("Extracting MobileNet features...")

with torch.no_grad():

    for i, row in enumerate(test_rows):

        image_path = DATA_DIR / row["path"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 64, 64)

        # Run through conv blocks only (not the final classifier)
        feature_maps = model.features(image_tensor)  # output shape: (1, 1280, 2, 2)

        # Collapse the 2x2 spatial maps to 1x1 by averaging → (1, 1280, 1, 1)
        features = torch.nn.functional.adaptive_avg_pool2d(
            feature_maps, (1, 1)
        )

        # Flatten to a 1D vector of 1280 numbers
        features = features.flatten(1)

        features_list.append(features.cpu().numpy()[0])
        labels_list.append(int(row["label"]))

        # Print progress every 250 images so we know the script isn't frozen
        if (i + 1) % 250 == 0:
         print(f"Processed {i + 1}/1000 images")


# Stack all 1000 feature vectors into a 2D numpy array: shape (1000, 1280)
features = np.array(features_list)
labels = np.array(labels_list)

print("Feature shape:", features.shape)


# Step 6: Run t-SNE to compress 1280 dimensions → 2 dimensions
# perplexity=30: roughly how many neighbours each point considers (30 is a common default)
# init="pca": start from PCA coordinates for a more stable result
# random_state=42: reproducible layout
print("Running t-SNE. This may take several minutes on CPU...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto",
)

# fit_transform takes (1000, 1280) and returns (1000, 2)
embeddings = tsne.fit_transform(features)

print("t-SNE completed!")


# Step 7: Plot the 2D embeddings, coloured by class
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

plt.figure(figsize=(12, 9))

for class_id, class_name in enumerate(class_names):
    # Get the indices of all images belonging to this class
    mask = labels == class_id

    # Scatter plot: x and y are the 2 t-SNE dimensions; each class gets its own colour
    plt.scatter(
        embeddings[mask, 0],
        embeddings[mask, 1],
        label=class_name,
        s=8,       # small dot size (1000 dots on one plot)
        alpha=0.7, # slightly transparent so overlapping dots are visible
    )

plt.title("t-SNE Visualization of MobileNetV2 Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(markerscale=2, fontsize=8)
plt.tight_layout()


# Step 8: Save the figure
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

output_path = FIGURES_DIR / "mobilenet_v2_tsne.png"

plt.savefig(output_path, dpi=120)
plt.show()

print("t-SNE figure saved:", output_path)
