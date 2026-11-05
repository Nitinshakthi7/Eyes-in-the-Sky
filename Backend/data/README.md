# Data

The dataset is **not** in git (it is ignored on purpose).

To get it, activate your venv and run from the repo root:

    python check_setup.py --name YourName --download

This downloads EuroSAT (RGB version, 27,000 images in 10 classes) with torchvision into `data/eurosat/`.
Source: https://github.com/phelber/eurosat (the original EuroSAT authors). torchvision downloads the same images from a copy on HuggingFace (about 94 MB).

If the download stops half way, delete the folder `data/eurosat` and run the command again (the script does not re-check a half-finished folder).

`data/splits.csv` (the fixed 70/15/15 split) is kept in git so everyone uses the same split.
