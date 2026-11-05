
"""gradcam.py - where did the model look?

Part E: Grad-CAM explainability.
"""

import torch
import torch.nn.functional as F


def grad_cam(model, image, layer):
    """Generate a Grad-CAM heatmap for one image.

    Args:
        model: trained classification model
        image: tensor shaped (1, 3, H, W)
        layer: target convolutional layer

    Returns:
        heatmap: tensor shaped (H, W), values between 0 and 1
        predicted_class: integer predicted class
    """

    model.eval()

    activations = None
    gradients = None

    # Save feature maps from the target layer
    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    # Save gradients flowing through the target layer
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()

        # Forward pass
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

        # Backpropagate the predicted class score
        score = output[0, predicted_class]
        score.backward()

        # Average gradients across spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Keep only positive importance
        cam = F.relu(cam)

        # Resize heatmap to original image dimensions
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to 0-1
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu(), predicted_class

    finally:
        forward_handle.remove()
        backward_handle.remove()
