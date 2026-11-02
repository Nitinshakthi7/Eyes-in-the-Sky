"""dataset.py - load EuroSAT, split it, augment it.

Owner: Shrivalli
Part:  A (Data + EDA + fixed split)

Will contain (one line per future function):
  load_eurosat(root)                   - load the EuroSAT images with torchvision
  make_split(seed)                     - fixed 70/15/15 stratified split, saved to data/splits.csv
  augment()                            - 90-degree rotations + flips (label-safe for satellite images)

TODO
"""
