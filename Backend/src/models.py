"""Models for the project.

Part B: Tejash builds the scratch CNN here.
Part C: Rachana adds the transfer-learning builders here.
"""

import torch.nn as nn
from torchvision import models


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


def build_transfer_model(name: str, freeze: bool = True):
    """Build a pretrained ImageNet model adapted for 10 EuroSAT classes.

    name:   'resnet18' or 'mobilenet_v2'
    freeze: True keeps the backbone fixed so only the new last layer trains;
            False lets all layers fine-tune.
    """
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported transfer model {name!r}; expected 'resnet18' or 'mobilenet_v2'")

    if freeze:
        # Freeze all pretrained weights first.
        for param in model.parameters():
            param.requires_grad = False

    # Replace the 1000-class ImageNet head with a new 10-class linear layer.
    # A new nn.Linear has requires_grad=True by default, so it always learns.
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif name == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    return model
