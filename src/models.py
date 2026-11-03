"""Models for the project.

Part B: Tejash builds the scratch CNN here.
Part C: Rachana adds the transfer-learning builders here.
"""

import torch.nn as nn


def build_scratch_cnn(
    num_classes=10,
    use_batchnorm=True,
    depth=3,
    dropout=0.0,
):
    """Build our CNN for 64x64 RGB images.

    The default settings are the baseline: 3 blocks with 32, 64 and 128 filters.
    Change one option at a time to make an ablation.
    """
    if depth not in (2, 3):
        raise ValueError("depth must be 2 or 3")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    filter_counts = [32, 64, 128][:depth]
    layers = []
    input_channels = 3

    for output_channels in filter_counts:
        # Each block finds image features, then halves the image width and height.
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))
        input_channels = output_channels

    # Keep one value per feature map, whatever the final image size is.
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(input_channels, num_classes))

    return nn.Sequential(*layers)
