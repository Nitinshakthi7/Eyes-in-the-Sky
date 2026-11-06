"""models.py - the neural network architectures used in this project.

Part B: Tejash builds the scratch CNN here.
Part C: Rachana adds the transfer-learning builders here.

This file has two types of models:
  1. A CNN we designed ourselves from scratch (build_scratch_cnn)
  2. Pre-trained models from ImageNet that we adapt for our 10 classes (build_transfer_model)
"""

import torch.nn as nn
from torchvision import models


def build_scratch_cnn(
    num_classes=10,
    use_batchnorm=True,
    depth=3,
    dropout=0.0,
):
    """Build our own CNN for 64x64 RGB satellite images.

    This is the model we designed ourselves (Part B). It uses convolutional blocks —
    each block scans the image for visual patterns (edges, textures, shapes) and then
    shrinks the image in half. After all blocks, we collapse the result to one value
    per feature and pass it to a final layer that picks one of the 10 land-use classes.

    Parameters:
      num_classes:   how many categories to predict (10 for EuroSAT)
      use_batchnorm: if True, adds BatchNorm after each convolution. This stabilises
                     training and usually improves accuracy. Set to False for the ablation.
      depth:         how many convolution blocks (2 or 3). We test both to see which is better.
      dropout:       randomly zeros some neurons during training to prevent overfitting.
                     0.0 means no dropout (the default for most of our runs).

    Default architecture (3 blocks):
      Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → image goes from 64x64 to 32x32
      Block 2: Conv(32→64) + BN + ReLU + MaxPool  → 32x32 to 16x16
      Block 3: Conv(64→128)+ BN + ReLU + MaxPool  → 16x16 to 8x8
      Then: AdaptiveAvgPool → 1x1, Flatten → Linear(128→10)
    """
    # Validate inputs before building anything
    if depth not in (2, 3):
        raise ValueError("depth must be 2 or 3")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be between 0 and 1")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    # For depth=3: [32, 64, 128]. For depth=2: [32, 64].
    # Each number is how many filters (feature detectors) that block uses.
    filter_counts = [32, 64, 128][:depth]
    layers = []
    input_channels = 3  # RGB images have 3 colour channels

    for output_channels in filter_counts:
        # --- One convolution block ---
        # Conv2d: scans the image with small 3x3 filters to find visual patterns.
        # padding=1 keeps the image the same size after convolution.
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))

        # BatchNorm: normalises the activations so training is smoother and faster.
        # Only added if use_batchnorm=True (we test with/without it as an ablation).
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(output_channels))

        # ReLU: the activation function. It just replaces any negative value with 0.
        # Without a non-linearity like this, stacking layers would do nothing useful.
        layers.append(nn.ReLU())

        # MaxPool: takes the maximum in each 2x2 region, halving the image dimensions.
        # This shrinks the image progressively while keeping the strongest features.
        layers.append(nn.MaxPool2d(kernel_size=2))

        input_channels = output_channels  # next block's input = this block's output channels

    # AdaptiveAvgPool: collapses whatever spatial size is left down to a single 1x1 value
    # per feature map. This means the final layer always gets a fixed-size input regardless
    # of the image size going in.
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())  # turn the (N, 128, 1, 1) tensor into a flat (N, 128) vector

    # Optionally drop some activations randomly during training to avoid overfitting
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))

    # Final classification layer: maps the 128 (or 64) features to one score per class
    layers.append(nn.Linear(input_channels, num_classes))

    # nn.Sequential chains all the layers into one callable model object
    return nn.Sequential(*layers)


def build_transfer_model(name: str, freeze: bool = True):
    """Build a pre-trained model adapted for our 10 EuroSAT land-use classes.

    "Transfer learning" means we start with a model already trained on 1 million
    ImageNet images, so it already knows how to detect edges, textures, and shapes.
    We just replace its final prediction layer (which outputs 1000 ImageNet classes)
    with a new one that outputs 10 (our EuroSAT classes).

    Parameters:
      name:   which backbone to use — "resnet18" or "mobilenet_v2"
      freeze: if True, we lock the pre-trained backbone weights and only train the
              new 10-class head. This is fast but less accurate.
              If False, all layers are free to update (fine-tuning). This is slower
              but usually gives better accuracy.

    Why these two models?
      ResNet-18:    a classic 18-layer network. Good baseline for transfer learning.
      MobileNetV2:  designed to be small and fast. Works well on our 64x64 images.
    """
    # Load the model with its official ImageNet weights (downloaded from PyTorch's servers)
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported transfer model {name!r}; expected 'resnet18' or 'mobilenet_v2'")

    if freeze:
        # Lock all existing weights — they will NOT change during training.
        # Only the new final layer (added below) will learn.
        for param in model.parameters():
            param.requires_grad = False

    # Replace the original 1000-class final layer with a fresh 10-class one.
    # A new nn.Linear always starts with requires_grad=True, so it will always learn
    # even when the rest of the model is frozen.
    if name == "resnet18":
        # resnet18.fc is the last fully-connected layer
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif name == "mobilenet_v2":
        # mobilenet_v2.classifier[1] is the last linear layer in its classifier block
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    return model
