"""gradcam_sanity_check.py - is Grad-CAM telling the truth on our model?

Part E control experiment (added by Nitin with Claude, because the confidence-drop experiment alone has no control).
Uses 200 test images: the first 20 of every class in the order of data/splits.csv (a fixed rule).

Check 1 - where is the heat? Which quarter (quadrant) of the 64 x 64 tile is hottest in each Grad-CAM map?
Check 2 - hide a patch of 25% of the pixels with the average colour and measure the fall in confidence of the predicted class:
          the HOTTEST 25%, the COLDEST 25%, and a random 32 x 32 square. If Grad-CAM were meaningful, hiding the hottest
          patch should hurt clearly more than hiding the coldest.

Run from the repo folder:  python -m src.gradcam_sanity_check      (CPU, about 1 minute)
Writes results/gradcam_sanity_check.csv (one row per image).
"""

import numpy as np
import pandas as pd
import torch

from src.dataset import EuroSATDataset
from src.gradcam import grad_cam
from src.train import build_model
from src.utils import MODELS_DIR, RESULTS_DIR

IMAGES_PER_CLASS = 20
HIDE = 1024                                   # 25% of the 64 x 64 = 4096 pixels
QUADRANTS = {(0, 0): "top-left", (0, 1): "top-right", (1, 0): "bottom-left", (1, 1): "bottom-right"}


def painted(batch, flat_ids):
    out = batch.clone().reshape(1, 3, -1)
    out[:, :, flat_ids] = 0.0                 # 0 after normalisation = the average colour of the training data
    return out.reshape(1, 3, 64, 64)


def main():
    model, _ = build_model("mobilenet_v2_full")
    state = torch.load(MODELS_DIR / "mobilenet_v2_full_frac1_seed0.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    data = EuroSATDataset("test")
    generator = torch.Generator().manual_seed(0)
    rows = []
    for label in range(10):
        for index in data.df.index[data.df["label"] == label][:IMAGES_PER_CLASS]:
            batch = data[int(index)][0].unsqueeze(0)
            heat, predicted = grad_cam(model, batch, model.features[-1])
            h = heat.numpy()
            quarters = np.array([[h[:32, :32].mean(), h[:32, 32:].mean()], [h[32:, :32].mean(), h[32:, 32:].mean()]])
            flat = heat.flatten()
            top, left = torch.randint(0, 33, (2,), generator=generator).tolist()
            square = ((torch.arange(32)[:, None] + top) * 64 + (torch.arange(32)[None, :] + left)).flatten()
            patches = {"hottest": flat.topk(HIDE).indices, "coldest": (-flat).topk(HIDE).indices, "random_square": square}
            with torch.no_grad():
                before = torch.softmax(model(batch), dim=1)[0, predicted].item()
                row = {"path": data.df.loc[index, "path"], "class_name": data.df.loc[index, "class_name"],
                       "hottest_quadrant": QUADRANTS[np.unravel_index(quarters.argmax(), (2, 2))]}
                for name, ids in patches.items():
                    probs = torch.softmax(model(painted(batch, ids)), dim=1)[0]
                    row[f"drop_{name}"] = before - probs[predicted].item()
                    row[f"changed_{name}"] = int(probs.argmax().item() != predicted)
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "gradcam_sanity_check.csv", index=False)

    n = len(table)
    print(f"Images tested: {n}")
    print("Check 1 - hottest quadrant of the Grad-CAM map:")
    for name in QUADRANTS.values():
        count = int((table["hottest_quadrant"] == name).sum())
        print(f"  {name:13s} {count:3d}  ({count / n:.0%})")
    print("Check 2 - hide 25% of the pixels:")
    for name in ("hottest", "coldest", "random_square"):
        print(f"  {name:14s} mean confidence drop {table[f'drop_{name}'].mean():.3f} | prediction changed for {int(table[f'changed_{name}'].sum())} of {n}")
    hot, cold = table["drop_hottest"], table["drop_coldest"]
    print(f"  hottest hurt MORE than coldest for {int((hot > cold).sum())} images, LESS for {int((hot < cold).sum())}")


if __name__ == "__main__":
    main()
