
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models import build_transfer_model
from gradcam import grad_cam
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

device = torch.device("cpu")

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

splits = pd.read_csv(Path("data/splits.csv"))
test_df = splits[splits["split"] == "test"].reset_index(drop=True)
wrong_df = pd.read_csv(RESULTS_DIR / "misclassified_images.csv")

samples = []

for class_name in class_names:
    mistakes = wrong_df[wrong_df["actual_class"] == class_name]
    if len(mistakes) > 0:
        row = mistakes.iloc[0]
        image_path = DATA_DIR / row["path"]
        samples.append((
            class_name,
            Image.open(image_path).convert("RGB"),
            False
        ))

correct_found = set()

for _, row in test_df.iterrows():
    class_name = row["class_name"]

    if class_name in correct_found:
        continue

    image_path = DATA_DIR / row["path"]
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor).argmax(1).item()

    if prediction == row["label"]:
        samples.append((class_name, image, True))
        correct_found.add(class_name)

    if len(correct_found) == 10:
        break

fig, axes = plt.subplots(
    len(samples), 3,
    figsize=(12, len(samples) * 2.8)
)

if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, (class_name, image, is_correct) in enumerate(samples):
    input_tensor = transform(image).unsqueeze(0)

    heatmap, predicted_class = grad_cam(
        model,
        input_tensor,
        layer=model.features[-1]
    )

    image_np = np.array(image.resize((64, 64)))

    axes[i, 0].imshow(image_np)
    axes[i, 0].set_title(
        f"{class_name} - {'Correct' if is_correct else 'Wrong'}"
    )
    axes[i, 0].axis("off")

    axes[i, 1].imshow(heatmap, cmap="jet")
    axes[i, 1].set_title("Grad-CAM")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(image_np)
    axes[i, 2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[i, 2].set_title(f"Predicted: {class_names[predicted_class]}")
    axes[i, 2].axis("off")

plt.tight_layout()

output = RESULTS_DIR / "figures" / "gradcam_all_classes.png"
plt.savefig(output, dpi=150)
plt.close()

print("Grad-CAM saved:", output)
print("Images included:", len(samples))
print("Correct examples:", len(correct_found))
print("Misclassified examples:", len(samples) - len(correct_found))
