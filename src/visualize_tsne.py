"""visualize_tsne.py - MobileNetV2 feature visualization."""
import os

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


# 1. Device
device = torch.device("cpu")


# 2. Load trained model
model = build_transfer_model("mobilenet_v2", freeze=False)

checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"

model.load_state_dict(
    torch.load(checkpoint, map_location=device, weights_only=True)
)

model.to(device)
model.eval()


# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 4. Read test images using Python CSV reader
split_file = DATA_DIR / "splits.csv"

test_rows = []

with open(split_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["split"] == "test":
            test_rows.append(row)

print("Total test images:", len(test_rows))


# Randomly select 1000 test images (fixed seed 42)
random.Random(42).shuffle(test_rows)
test_rows = test_rows[:1000]


# 5. Extract features
features_list = []
labels_list = []

print("Extracting MobileNet features...")

with torch.no_grad():

    for i, row in enumerate(test_rows):

        image_path = DATA_DIR / row["path"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        feature_maps = model.features(image_tensor)

        features = torch.nn.functional.adaptive_avg_pool2d(
            feature_maps, (1, 1)
        )

        features = features.flatten(1)

        features_list.append(features.cpu().numpy()[0])
        labels_list.append(int(row["label"]))

        if (i + 1) % 250 == 0:
         print(f"Processed {i + 1}/1000 images")


features = np.array(features_list)
labels = np.array(labels_list)

print("Feature shape:", features.shape)


# 6. Run t-SNE
print("Running t-SNE. This may take several minutes on CPU...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto",
)

embeddings = tsne.fit_transform(features)

print("t-SNE completed!")


# 7. Plot
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

plt.figure(figsize=(12, 9))

for class_id, class_name in enumerate(class_names):

    mask = labels == class_id

    plt.scatter(
        embeddings[mask, 0],
        embeddings[mask, 1],
        label=class_name,
        s=8,
        alpha=0.7,
    )

plt.title("t-SNE Visualization of MobileNetV2 Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(markerscale=2, fontsize=8)
plt.tight_layout()


# 8. Save
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

output_path = FIGURES_DIR / "mobilenet_v2_tsne.png"

plt.savefig(output_path, dpi=120)
plt.show()

print("t-SNE figure saved:", output_path)
