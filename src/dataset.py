"""dataset.py - load EuroSAT, split it, augment it.

Owner: Srivalli
Part:  A (Data + EDA + fixed split)

Contains:
  load_eurosat(root)                   - load the EuroSAT images with torchvision
  make_split(seed)                     - fixed 70/15/15 stratified split, saved to data/splits.csv
  check_split(df)                      - prove there is no leakage (sizes, class balance, no shared files)
  augment()                            - 90-degree rotations + flips (label-safe for satellite images)
  EuroSATDataset(split, transform)     - reads splits.csv, returns (image, label)
  get_loader(split, fraction, seed, augment, batch_size)  - the DataLoader everyone else calls

Make the split (venv on, from the repo folder; only needed once):  python -m src.dataset
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils import DATA_DIR, SPLITS_CSV

# Team setting: ImageNet numbers for ALL models, so the comparison is fair.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_SIZES = {"train": 18900, "val": 4050, "test": 4050}  # 70/15/15 of 27,000


def load_eurosat(root=DATA_DIR):
    """Load the EuroSAT images that check_setup.py downloaded into data/eurosat/2750."""
    # .samples is a list of (file path, label) pairs, .classes is the list of class names
    return datasets.EuroSAT(root=str(root), download=False)


def make_split(seed=42):
    """Make the fixed 70/15/15 split and save it to data/splits.csv."""
    eurosat = load_eurosat()

    # One row per image. We save the path from the data folder with "/" so the
    # same file works on every laptop (the full path is different on each one).
    rows = []
    for path, label in eurosat.samples:
        relative_path = Path(path).relative_to(DATA_DIR).as_posix()
        rows.append((relative_path, label, eurosat.classes[label]))
    df = pd.DataFrame(rows, columns=["path", "label", "class_name"])

    # stratify= keeps the same share of each class in every part.
    # First cut: 70% train, 30% rest.
    train_idx, rest_idx = train_test_split(
        df.index, train_size=0.7, stratify=df["label"], random_state=seed)
    # Second cut: half of the rest is validation, half is test (15% + 15%).
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=0.5, stratify=df.loc[rest_idx, "label"], random_state=seed)

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # lineterminator: same line endings on every laptop, so git sees no fake changes
    df.to_csv(SPLITS_CSV, index=False, lineterminator="\n")
    return df


def check_split(df):
    """Prove there is no leakage. Stops with an error if something is wrong."""
    # 1. The split sizes are the expected ones.
    sizes = df["split"].value_counts().to_dict()
    assert sizes == SPLIT_SIZES, f"wrong split sizes: {sizes}"

    # 2. No file appears twice, so no file can be in two splits.
    assert df["path"].is_unique, "a file appears in more than one row"

    # 3. Every class has the same share in each split (at most 1 image off from exact rounding).
    counts = pd.crosstab(df["class_name"], df["split"])
    total_per_class = counts.sum(axis=1)
    for split, size in SPLIT_SIZES.items():
        expected = total_per_class * size / len(df)
        assert ((counts[split] - expected).abs() <= 1).all(), f"{split} is not stratified"

    print("Leakage check: OK")
    print(f"  split sizes: {sizes}")
    print("  no file appears in two splits; every class keeps its share in every split")
    return counts


def augment():
    """Random 90-degree rotation and flips. Only used on training images."""
    # A satellite looks straight down, so there is no "up": a rotated or flipped
    # field is still a field. The angles are exact, so no pixels get blurred.
    return transforms.Compose([
        transforms.RandomChoice([transforms.RandomRotation((angle, angle))
                                 for angle in (0, 90, 180, 270)]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])


def make_transform(use_augment=False):
    """Turn a PIL image into a normalised tensor, with augmentation if asked."""
    steps = [augment()] if use_augment else []
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class EuroSATDataset(Dataset):
    """One part of the fixed split: gives back (image, label)."""

    def __init__(self, split, transform=None):
        df = pd.read_csv(SPLITS_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform if transform is not None else make_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(DATA_DIR / row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def get_loader(split, fraction=1.0, seed=0, augment=False, batch_size=64, num_workers=0):
    """Make the DataLoader for "train", "val" or "test".

    fraction: share of the TRAIN split to use (stratified). val and test never shrink.
    seed:     picks the subset and the shuffle order, so a run can be repeated.
    augment:  rotations and flips, allowed only for the train split.
    """
    if split != "train" and (fraction != 1.0 or augment):
        raise ValueError("fraction and augment are only for the train split")

    data = EuroSATDataset(split, transform=make_transform(augment))

    if fraction < 1.0:
        # Keep the same class shares in the smaller training set.
        keep, _ = train_test_split(
            data.df.index, train_size=fraction, stratify=data.df["label"], random_state=seed)
        data.df = data.df.loc[sorted(keep)].reset_index(drop=True)

    # Only the training data is shuffled. The generator makes the order repeatable.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=(split == "train"),
                      generator=generator, num_workers=num_workers)


if __name__ == "__main__":
    # Make data/splits.csv and check it.
    check_split(make_split())
