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

# These two constants are the standard "ImageNet normalisation" numbers.
# Every model in this project uses them so comparisons between models are fair —
# the input data is always scaled the same way.
# mean: average pixel value (per colour channel) across the ImageNet dataset
# std:  standard deviation of pixel values across the ImageNet dataset
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# EuroSAT has 27,000 images total. We split them 70% train / 15% val / 15% test.
# This dictionary lets us check those sizes haven't changed accidentally.
SPLIT_SIZES = {"train": 18900, "val": 4050, "test": 4050}  # 70/15/15 of 27,000


def load_eurosat(root=DATA_DIR):
    """Load the EuroSAT dataset from disk (already downloaded by check_setup.py).

    torchvision's EuroSAT class reads all the image file paths and their class labels.
    We use download=False because check_setup.py already handled the download.
    Returns a dataset object where .samples is a list of (file path, label number) pairs
    and .classes is the list of human-readable class names like "Forest", "Highway" etc.
    """
    return datasets.EuroSAT(root=str(root), download=False)


def make_split(seed=42):
    """Create the fixed 70/15/15 train/val/test split and save it to data/splits.csv.

    We do this ONCE (not every run) so every team member uses the exact same split.
    Using stratify= means every class has the same proportion in each part —
    we don't accidentally put all the "Highway" images in train, for example.

    The CSV has columns: path, label, class_name, split
    We store relative paths (not absolute) so the CSV works on every laptop.
    """
    eurosat = load_eurosat()

    # Build a table with one row per image: its relative path, numeric label, and class name
    rows = []
    for path, label in eurosat.samples:
        # Convert the full system path to a relative path starting from data/
        # We use as_posix() to always write "/" separators (works on all operating systems)
        relative_path = Path(path).relative_to(DATA_DIR).as_posix()
        rows.append((relative_path, label, eurosat.classes[label]))
    df = pd.DataFrame(rows, columns=["path", "label", "class_name"])

    # First cut: 70% goes to train, the remaining 30% goes into a "rest" pile
    # stratify=df["label"] ensures each class stays proportional across the split
    train_idx, rest_idx = train_test_split(
        df.index, train_size=0.7, stratify=df["label"], random_state=seed)

    # Second cut: split the 30% "rest" evenly into val (15%) and test (15%)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=0.5, stratify=df.loc[rest_idx, "label"], random_state=seed)

    # Label each row with which split it belongs to
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Save to CSV. lineterminator="\n" gives consistent line endings on all operating systems
    # so git doesn't flag fake changes when different team members open the file.
    df.to_csv(SPLITS_CSV, index=False, lineterminator="\n")
    return df


def check_split(df):
    """Verify the split has no data leakage and the right sizes.

    "Leakage" means the same image appears in both train and test — that would
    make the accuracy look better than it really is. This function crashes with
    a clear error message if anything is wrong.
    """
    # 1. Check the train/val/test counts match what we expect
    sizes = df["split"].value_counts().to_dict()
    assert sizes == SPLIT_SIZES, f"wrong split sizes: {sizes}"

    # 2. Every file path must be unique — if a file appears twice, it could be in two splits
    assert df["path"].is_unique, "a file appears in more than one row"

    # 3. Each class must appear roughly the same proportion in every split
    # (at most 1 image off from exact due to rounding)
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
    """Return a set of random flips and rotations to apply to training images only.

    Why these transformations? A satellite always looks straight down, so there is no
    natural "up" direction. A forest rotated 90 degrees is still a forest. This means
    we can safely rotate and flip any image and it still has the correct label.
    We use exact 90-degree angles (not arbitrary angles) so no pixels get blurred.

    We only apply this during TRAINING, never during validation or testing,
    because we want those sets to stay as close to real-world input as possible.
    """
    return transforms.Compose([
        # Randomly choose one of 0°, 90°, 180°, or 270° rotation
        transforms.RandomChoice([transforms.RandomRotation((angle, angle))
                                 for angle in (0, 90, 180, 270)]),
        transforms.RandomHorizontalFlip(p=0.5),   # flip left-right 50% of the time
        transforms.RandomVerticalFlip(p=0.5),     # flip upside-down 50% of the time
    ])


def make_transform(use_augment=False):
    """Build the full image pre-processing pipeline for a split.

    Every image must go through the same steps before the model sees it:
    1. (Optional) Random flips/rotations for training data
    2. Convert the PIL image to a PyTorch tensor (scales pixel values from 0-255 to 0.0-1.0)
    3. Normalise using ImageNet mean and std — this centres the data around 0
       which helps the model learn faster and converge more reliably.
    """
    steps = [augment()] if use_augment else []
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class EuroSATDataset(Dataset):
    """A PyTorch Dataset that serves images from one split (train, val, or test).

    PyTorch needs a Dataset object to understand how to fetch individual examples.
    It needs two things: __len__ (how many images are there?) and __getitem__
    (give me image number i).

    We read splits.csv to know which images belong to this split, then load
    each image from disk on demand (one at a time, as the training loop requests it).
    """

    def __init__(self, split, transform=None):
        # Read the full split table and keep only the rows for this split (e.g. "train")
        df = pd.read_csv(SPLITS_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Use the provided transform pipeline, or build a default one (no augmentation)
        self.transform = transform if transform is not None else make_transform()

    def __len__(self):
        # How many images are in this split
        return len(self.df)

    def __getitem__(self, i):
        # Load image number i from disk, apply the transform, and return (image tensor, label number)
        row = self.df.iloc[i]
        image = Image.open(DATA_DIR / row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])


def get_loader(split, fraction=1.0, seed=0, augment=False, batch_size=64, num_workers=0):
    """Build and return a DataLoader — the object the training loop iterates over.

    A DataLoader wraps a Dataset and feeds images in batches (groups of 64 by default).
    It handles shuffling and can work in parallel with multiple CPU workers.

    split:      "train", "val", or "test"
    fraction:   for training only — use e.g. 0.1 to train on just 10% of the data
                (used in the data-efficiency study to see how accuracy changes with less data)
    seed:       ensures the same subset and shuffle order every time, so runs are repeatable
    augment:    apply random flips/rotations — training only, never val or test
    batch_size: how many images to process at once (64 is the team standard)
    """
    # Safety check: fraction and augment only make sense for the training split
    if split != "train" and (fraction != 1.0 or augment):
        raise ValueError("fraction and augment are only for the train split")

    data = EuroSATDataset(split, transform=make_transform(augment))

    if fraction < 1.0:
        # Randomly pick a stratified subset of the training images
        # (stratified = same class proportions as the full set, just fewer images)
        keep, _ = train_test_split(
            data.df.index, train_size=fraction, stratify=data.df["label"], random_state=seed)
        data.df = data.df.loc[sorted(keep)].reset_index(drop=True)

    # Only the training set is shuffled. The Generator with a fixed seed makes the
    # shuffle order repeatable — same seed = same image order every run.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=(split == "train"),
                      generator=generator, num_workers=num_workers)


if __name__ == "__main__":
    # Run once to create data/splits.csv, then verify it has no leakage.
    # Command: python -m src.dataset
    check_split(make_split())
