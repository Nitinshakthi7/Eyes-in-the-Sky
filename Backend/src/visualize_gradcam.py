import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models import build_transfer_model
from src.gradcam import grad_cam
from src.utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

# Device
device = torch.device("cpu")

# Load trained MobileNetV2
model = build_transfer_model("mobilenet_v2", freeze=False)

checkpoint = MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Read misclassified images
df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

# Select first 4 mistakes
df = df.head(4).reset_index(drop=True)

class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake"
]

fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i, row in df.iterrows():

    image_path = DATA_DIR / row["path"]
    original = Image.open(image_path).convert("RGB")

    image_tensor = transform(original).unsqueeze(0).to(device)

    # Generate Grad-CAM
    heatmap, predicted_class = grad_cam(
        model,
        image_tensor,
        model.features[-1]
    )

    heatmap = heatmap.numpy()

    # Original image
    axes[i, 0].imshow(original)
    axes[i, 0].set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}"
    )
    axes[i, 0].axis("off")

    # Heatmap overlay
    axes[i, 1].imshow(original)
    axes[i, 1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[i, 1].set_title("Grad-CAM")
    axes[i, 1].axis("off")

plt.tight_layout()

output_path = RESULTS_DIR / "figures" / "gradcam_examples.png"
plt.savefig(output_path, dpi=120, bbox_inches="tight")
plt.show()

print("Grad-CAM figure saved:", output_path)
