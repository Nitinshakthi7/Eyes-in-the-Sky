"""gradcam.py - where did the model look when it made a decision?

Part E: Grad-CAM explainability.

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that highlights
which parts of an image the model paid most attention to when making its prediction.
It does this by tracking how much each part of the last convolutional layer contributed
to the final score — the hotter the region, the more important it was.

This file is the core Grad-CAM engine. Other files (visualize_gradcam.py,
gradcam_all_classes.py etc.) call this to generate heatmaps.
"""

import torch
import torch.nn.functional as F


def grad_cam(model, image, layer):
    """Generate a Grad-CAM heatmap showing where the model looked for one image.

    How it works:
    1. We attach "hooks" to the target conv layer to record what it outputs (activations)
       and what gradients flow back through it during backprop.
    2. We do a forward pass to get the model's prediction.
    3. We backpropagate the score for the predicted class.
    4. We average the gradients across the spatial dimensions — this gives one weight
       per feature map, telling us how important each filter was for the prediction.
    5. We take a weighted sum of the activation maps using those weights.
    6. We apply ReLU to keep only positive contributions (negative = hurts the prediction).
    7. We resize the result back to the original image size and normalise to [0, 1].

    Args:
        model:  the trained classification model (must be in eval mode)
        image:  a single image tensor, shape (1, 3, H, W) — batch size of 1
        layer:  the target convolutional layer to look at (usually the last conv block)

    Returns:
        heatmap:         tensor of shape (H, W) with values between 0 and 1
                         (1 = most important region, 0 = least important)
        predicted_class: integer — the class the model predicted
    """
    model.eval()

    # These will be filled by the hooks we attach below
    activations = None
    gradients = None

    # --- Hook 1: forward hook ---
    # This is called automatically by PyTorch when the layer does its forward pass.
    # We save the layer's output (the feature maps) so we can use them later.
    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    # --- Hook 2: backward hook ---
    # This is called automatically by PyTorch during backpropagation.
    # We save the gradients flowing through this layer.
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register both hooks on the target layer
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()  # clear any old gradients from previous runs

        # Forward pass: run the image through the model to get class scores
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

        # Backpropagate just the score for the predicted class.
        # This computes "how does this layer's output affect the predicted class score?"
        score = output[0, predicted_class]
        score.backward()

        # Average the gradients over the spatial (height x width) dimensions.
        # Result: one weight per feature map (channel), shape (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps:
        # multiply each feature map by its importance weight, then add them all up
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: only keep regions that helped predict the class (positive values).
        # Negative values would mean "this region argues AGAINST this class" — we ignore those.
        cam = F.relu(cam)

        # Resize the small feature map (e.g. 8x8) back to the original image size (64x64)
        # so we can overlay it on the original image
        cam = F.interpolate(
            cam,
            size=image.shape[2:],  # target size = original image H x W
            mode="bilinear",        # smooth upscaling
            align_corners=False,
        )

        # Normalise to [0, 1]: 0 = unimportant, 1 = most important
        cam = cam[0, 0]              # remove batch and channel dims → shape (H, W)
        cam = cam - cam.min()        # shift minimum to 0
        cam = cam / (cam.max() + 1e-8)  # divide by max (add tiny value to avoid dividing by 0)

        return cam.detach().cpu(), predicted_class

    finally:
        # Always remove the hooks after we're done, even if an error occurred.
        # Leaving hooks attached would slow down every future forward/backward pass.
        forward_handle.remove()
        backward_handle.remove()
