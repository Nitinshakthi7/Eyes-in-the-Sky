"""server.py - the web application backend for Eyes in the Sky.

Owner: built with AI assistance and disclosed in docs/ai_usage.md.
Part:  Website / Demo application.

What this file does:
This script starts a local web server using FastAPI.
It connects our trained deep learning models to a web browser frontend:
  1. GET  /api/models   - tells the frontend which trained models are available to test.
  2. POST /api/predict  - takes an uploaded satellite image, predicts its land-use class,
                          calculates top-3 class probabilities, and generates a visual Grad-CAM heatmap overlay.
  3. GET  /api/metrics  - serves accuracy, confusion matrix, and data-efficiency study data for the dashboard.
  4. Static files       - serves the HTML, CSS, and JavaScript user interface directly from frontend/.

How to run it:
  1. Make sure your virtual environment is activated.
  2. Run from the Backend directory:
     python -m app.server
  3. Open your browser and visit:
     http://127.0.0.1:8000/
"""

import base64
import io
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import grad_cam
from src.train import build_model
from src.utils import METRICS_CSV, MODELS_DIR, RESULTS_DIR, ROOT

# Path to the frontend files (HTML/CSS/JS) located one directory up in frontend/
FRONTEND = ROOT.parent / "frontend"

# The 10 EuroSAT land-use classes in the exact order the model was trained on
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Set a safe upload limit (15 Megabytes) so users don't crash the server with huge files
MAX_UPLOAD = 15_000_000

# Default model used for the overall evaluation metrics page (Part E)
HEADLINE = "mobilenet_v2_full_frac1_seed0"


def last_relu(net):
    """Find the final ReLU activation layer in our custom scratch CNN.

    Why do we need this?
    Grad-CAM needs to hook into the final convolutional layer of a network
    to see what spatial patterns the model detected right before making a decision.
    In our Sequential scratch CNN, the end of the last conv block is the last nn.ReLU.
    """
    return [m for m in net if isinstance(m, nn.ReLU)][-1]


# Available model choices for the web interface.
# Each entry maps model_id -> (Human readable title, function to locate the target layer for Grad-CAM)
CHOICES = {
    # For MobileNetV2, features[-1] is the final inverted residual block before the classifier
    "mobilenet_v2_full": ("MobileNetV2 (fully fine-tuned)", lambda net: net.features[-1]),
    # For ResNet-18, layer4[-1] is the final residual block
    "resnet18_full": ("ResNet-18 (fully fine-tuned)", lambda net: net.layer4[-1]),
    # For our scratch CNN, we find the last ReLU in the sequential stack
    "scratch_cnn": ("Scratch CNN (our own, 3 blocks)", last_relu),
}

# Image preprocessing pipeline for incoming web uploads.
# Why 64x64? All our models were trained on 64x64 EuroSAT tiles, so any uploaded picture
# must be resized to 64x64 and scaled using standard ImageNet mean and standard deviation.
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Global in-memory cache for models: model_id -> (loaded_pytorch_model, target_gradcam_layer)
# This prevents reloading heavy weights from disk every time someone clicks "Predict"
_loaded = {}

# Thread lock: PyTorch hooks in Grad-CAM attach to the shared network object.
# To prevent two simultaneous web requests from corrupting each other's gradients,
# we process predictions one at a time using this lock.
_lock = threading.Lock()


def available():
    """Check the models/ folder and return a list of model IDs whose weights exist on disk.

    Why do this dynamically?
    If someone has only trained MobileNetV2 so far, the website will only show MobileNetV2
    in the dropdown instead of crashing when someone picks an untrained model.
    """
    return [i for i in CHOICES if (MODELS_DIR / f"{i}_frac1_seed0.pth").exists()]


def get_model(model_id):
    """Load model weights into memory if not already cached, and return (model, target_layer).

    Why use CPU here?
    For a lightweight local demo server serving one image at a time, CPU inference
    takes less than 50 milliseconds and avoids requiring a dedicated GPU to run the website.
    """
    if model_id not in _loaded:
        net, _ = build_model(model_id)
        weights_file = MODELS_DIR / f"{model_id}_frac1_seed0.pth"
        net.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        net.eval()  # Set to evaluation mode (turns off dropout, fixes batchnorm)
        _loaded[model_id] = (net, CHOICES[model_id][1](net))
    return _loaded[model_id]


def overlay_png(heat, size):
    """Convert a raw 2D Grad-CAM heatmap into a transparent, colored PNG image.

    Why do this?
    The browser frontend needs an image it can easily place directly on top
    of the satellite photo. We apply the 'jet' colormap (blue = low, red = high focus)
    and adjust transparency so that colder areas are mostly see-through, letting the user
    see the original satellite features underneath the heat spots.

    Returns a base64 encoded data-URL string (e.g. 'data:image/png;base64,...').
    """
    width, height = size
    # Keep the image dimensions manageable (at most 512px) so the response stays small and fast
    scale = min(1.0, 512 / max(width, height))
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    # Upsample heatmap smoothly to match the user's image dimensions
    heat_resized = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize(target_size, Image.BILINEAR)) / 255.0

    # Apply 'jet' color palette: gives an RGBA float array of shape (height, width, 4)
    rgba = colormaps["jet"](heat_resized)

    # Set dynamic alpha (transparency): hotter regions are more opaque (visible), cooler areas are see-through
    rgba[..., 3] = 0.15 + 0.55 * heat_resized

    # Save to an in-memory byte buffer instead of saving temporary files to disk
    buffer = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buffer, format="PNG")

    # Encode as base64 string so the frontend <img> tag can display it immediately
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# Initialize the FastAPI web application
app = FastAPI(title="Eyes in the Sky")

# Enable Cross-Origin Resource Sharing (CORS) so the frontend can query the API
# even if served from a different port or local address during testing
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/models")
def api_models():
    """Endpoint: returns a list of trained models currently available to choose from."""
    return {"models": [{"id": i, "label": CHOICES[i][0]} for i in available()]}


@app.post("/api/predict")
def api_predict(image: UploadFile = File(...), model: str = Form(None)):
    """Endpoint: classify an uploaded satellite image tile and compute explainability heatmap.

    Steps:
      1. Check if model weights exist on the machine.
      2. Verify the uploaded file is valid and within size limits.
      3. Preprocess the image to 64x64 normalized tensor.
      4. Run Grad-CAM to find the most influential image regions.
      5. Run model forward pass to compute softmax probabilities.
      6. Return top predicted class, confidence %, top-3 predictions, and Grad-CAM overlay.
    """
    ids = available()
    if not ids:
        raise HTTPException(503, "no model weights found in Backend/models/ (see models/README.md)")

    model_id = model or ids[0]
    if model_id not in ids:
        raise HTTPException(400, f"unknown model {model_id!r}; available: {ids}")

    # Read image content with a safety size check
    data = image.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image is larger than 15 MB")

    # Validate that the file is actually a readable image
    try:
        picture = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "the uploaded file is not an image")

    # Prepare input tensor with batch dimension: (1, 3, 64, 64)
    batch = TRANSFORM(picture).unsqueeze(0)

    # Acquire lock so only one thread attaches Grad-CAM hooks to the PyTorch model at a time
    with _lock:
        net, layer = get_model(model_id)
        # Compute where the model looked
        heat, _ = grad_cam(net, batch, layer)
        with torch.no_grad():
            # Convert raw model logits to probabilities that sum to 1.0 (100%)
            probabilities = torch.softmax(net(batch), dim=1)[0]

    # Find the top 3 highest predicted classes
    top = probabilities.topk(3)

    return {
        "label": CLASSES[int(top.indices[0])],
        "confidence": float(top.values[0]),
        "top3": [{"label": CLASSES[int(i)], "prob": float(p)} for p, i in zip(top.values, top.indices)],
        "gradcam": overlay_png(heat.numpy(), picture.size),
    }


@app.get("/api/metrics")
def api_metrics():
    """Endpoint: serves overall test metrics and curves for the dashboard charts.

    Reads:
      - Confusion matrix CSV
      - Classification report CSV
      - Metrics CSV containing the data-efficiency curves
    """
    matrix = pd.read_csv(RESULTS_DIR / f"{HEADLINE}_confusion_matrix.csv", index_col=0)
    report = pd.read_csv(RESULTS_DIR / f"{HEADLINE}_classification_report.csv", index_col=0)
    runs = pd.read_csv(METRICS_CSV)

    # Filter to only the 3 canonical seeds (0, 1, 2) to match our report tables
    runs = runs[runs["seed"].isin([0, 1, 2])]
    runs["fraction"] = runs["fraction"].round(2)

    def curve(model_name):
        """Helper to extract mean accuracy across seeds for each training data fraction (5% to 100%)."""
        mean = runs[runs["model"] == model_name].groupby("fraction")["accuracy"].mean() * 100
        return [round(float(mean.loc[f]), 2) for f in (0.05, 0.1, 0.25, 0.5, 1.0)]

    return {
        # Overall accuracy = sum of diagonal (correct) divided by total samples
        "accuracy": round(100 * float(np.trace(matrix.values)) / float(matrix.values.sum()), 2),
        "macro_f1": round(100 * float(report.loc["macro avg", "f1-score"]), 2),
        "confusion": matrix.values.astype(int).tolist(),
        "efficiency": {
            "fractions": [5, 10, 25, 50, 100],
            "scratch": curve("scratch_cnn"),
            "transfer": curve("mobilenet_v2_full")
        },
    }


# Static frontend hosting configuration
# If the frontend/ directory is present, serve the user interface directly through FastAPI
if FRONTEND.is_dir():
    @app.get("/")
    def home():
        """Redirect the root URL to the landing presentation page."""
        return RedirectResponse("/Eyes_in_the_Sky_landing.html")

    @app.get("/app.html")
    def classifier_page(request: Request):
        """Serve the interactive classifier page.

        Why redirect with query parameter?
        The interactive web app defaults to demo mode unless pointed to this live server API.
        Adding ?api=<server_address> automatically connects the page to this running backend.
        """
        if "api" not in request.query_params:
            return RedirectResponse(f"/app.html?api={str(request.base_url).rstrip('/')}")
        return FileResponse(FRONTEND / "app.html")

    # Mount static assets (images, stylesheets, JS files) so the browser can load them
    app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")


if __name__ == "__main__":
    # Launch local Uvicorn web server listening on 127.0.0.1 port 8000
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
