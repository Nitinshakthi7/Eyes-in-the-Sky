"""server.py - the website backend (built with AI help, disclosed in the report).

Owner: none - built by AI after Gate 2 and disclosed in the report and in docs/ai_usage.md
Part:  website (an extra, not one of parts A-E)

One process serves the pages in ../frontend (Tejash) and the API that frontend/API.md describes:
  GET  /api/models    - the models whose weights are in models/ (file name: <model>_frac1_seed0.pth)
  POST /api/predict   - class, confidence, top 3 and a Grad-CAM overlay for one uploaded image
  GET  /api/metrics   - test accuracy, macro-F1, confusion matrix and the data-efficiency curves (from results/)

Run it from the Backend folder:  python -m app.server       then open  http://127.0.0.1:8000/
It works offline (nothing is downloaded) and listens on this computer only.
Check it:  python -m app.test_server
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

FRONTEND = ROOT.parent / "frontend"                 # Tejash's pages
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]    # EuroSAT folder order
MAX_UPLOAD = 15_000_000                              # bytes
HEADLINE = "mobilenet_v2_full_frac1_seed0"           # the model behind the metrics page (Part E)


def last_relu(net):
    """Last ReLU of the scratch CNN = end of its last convolution block."""
    return [m for m in net if isinstance(m, nn.ReLU)][-1]


# id -> (label for the page, how to find the layer Grad-CAM looks at)
CHOICES = {
    "mobilenet_v2_full": ("MobileNetV2 (fully fine-tuned)", lambda net: net.features[-1]),
    "resnet18_full": ("ResNet-18 (fully fine-tuned)", lambda net: net.layer4[-1]),
    "scratch_cnn": ("Scratch CNN (our own, 3 blocks)", last_relu),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),                     # the models were trained on 64 x 64 tiles
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_loaded = {}                                         # model id -> (network, Grad-CAM layer)
_lock = threading.Lock()                             # Grad-CAM puts hooks on the shared network: one request at a time


def available():
    return [i for i in CHOICES if (MODELS_DIR / f"{i}_frac1_seed0.pth").exists()]


def get_model(model_id):
    if model_id not in _loaded:
        net, _ = build_model(model_id)
        net.load_state_dict(torch.load(MODELS_DIR / f"{model_id}_frac1_seed0.pth", map_location="cpu", weights_only=True))
        net.eval()
        _loaded[model_id] = (net, CHOICES[model_id][1](net))
    return _loaded[model_id]


def overlay_png(heat, size):
    """Grad-CAM map (64 x 64, values 0 to 1) -> transparent PNG with the size of the uploaded image, as a data URL."""
    width, height = size
    scale = min(1.0, 512 / max(width, height))        # keep the answer small
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) / 255.0
    rgba = colormaps["jet"](heat)                      # (h, w, 4) floats
    rgba[..., 3] = 0.15 + 0.55 * heat                  # hot areas more opaque
    buffer = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


app = FastAPI(title="Eyes in the Sky")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   # API.md asks for CORS


@app.get("/api/models")
def api_models():
    return {"models": [{"id": i, "label": CHOICES[i][0]} for i in available()]}


@app.post("/api/predict")
def api_predict(image: UploadFile = File(...), model: str = Form(None)):
    ids = available()
    if not ids:
        raise HTTPException(503, "no model weights found in Backend/models/ (see models/README.md)")
    model_id = model or ids[0]
    if model_id not in ids:
        raise HTTPException(400, f"unknown model {model_id!r}; available: {ids}")
    data = image.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image is larger than 15 MB")
    try:
        picture = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "the uploaded file is not an image")

    batch = TRANSFORM(picture).unsqueeze(0)
    with _lock:
        net, layer = get_model(model_id)
        heat, _ = grad_cam(net, batch, layer)
        with torch.no_grad():
            probabilities = torch.softmax(net(batch), dim=1)[0]
    top = probabilities.topk(3)
    return {
        "label": CLASSES[int(top.indices[0])],
        "confidence": float(top.values[0]),
        "top3": [{"label": CLASSES[int(i)], "prob": float(p)} for p, i in zip(top.values, top.indices)],
        "gradcam": overlay_png(heat.numpy(), picture.size),
    }


@app.get("/api/metrics")
def api_metrics():
    matrix = pd.read_csv(RESULTS_DIR / f"{HEADLINE}_confusion_matrix.csv", index_col=0)
    report = pd.read_csv(RESULTS_DIR / f"{HEADLINE}_classification_report.csv", index_col=0)
    runs = pd.read_csv(METRICS_CSV)
    runs = runs[runs["seed"].isin([0, 1, 2])]          # the 3 seeds every setting has (same as the README table)
    runs["fraction"] = runs["fraction"].round(2)

    def curve(model_name):
        mean = runs[runs["model"] == model_name].groupby("fraction")["accuracy"].mean() * 100
        return [round(float(mean.loc[f]), 2) for f in (0.05, 0.1, 0.25, 0.5, 1.0)]

    return {
        "accuracy": round(100 * float(np.trace(matrix.values)) / float(matrix.values.sum()), 2),
        "macro_f1": round(100 * float(report.loc["macro avg", "f1-score"]), 2),
        "confusion": matrix.values.astype(int).tolist(),
        "efficiency": {"fractions": [5, 10, 25, 50, 100], "scratch": curve("scratch_cnn"), "transfer": curve("mobilenet_v2_full")},
    }


if FRONTEND.is_dir():
    @app.get("/")
    def home():
        return RedirectResponse("/Eyes_in_the_Sky_landing.html")

    @app.get("/app.html")
    def classifier_page(request: Request):
        # The page runs in demo mode (fake results) unless it is opened with ?api=<this server>. Add that for the user.
        if "api" not in request.query_params:
            return RedirectResponse(f"/app.html?api={str(request.base_url).rstrip('/')}")
        return FileResponse(FRONTEND / "app.html")

    app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")        # last: catches every other file


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
