"""test_server.py - automated test check for the website backend.

Why we use this file:
This file performs end-to-end testing of the FastAPI web application without needing pytest.
It simulates real browser HTTP requests and tests:
  1. GET  /api/models   - confirms the endpoint lists available model checkpoints.
  2. POST /api/predict  - sends a real satellite image from the test set, verifying that
                          predictions, confidence scores, and Grad-CAM overlays are returned properly.
  3. Error handling     - tests that sending invalid files or unsupported model names returns proper 400 errors.
  4. GET  /api/metrics  - verifies the confusion matrix and data-efficiency curves return valid numbers.
  5. HTML web routes    - ensures static HTML pages and redirects work properly.

How to run it (from the Backend folder):
  python -m app.test_server
"""

import io

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from app.server import CLASSES, app, available
from src.utils import DATA_DIR, SPLITS_CSV

# TestClient lets us simulate web requests directly against the FastAPI app
# without needing to start a live Uvicorn server in a separate terminal.
client = TestClient(app)


def main():
    # --- Check 1: Ensure at least one trained model exists in models/ ---
    # The server cannot make predictions if no trained weights exist yet.
    ids = available()
    assert ids, "put at least one <model>_frac1_seed0.pth in models/"

    # --- Check 2: Verify the /api/models endpoint ---
    # The returned list of models should match the checkpoints found on disk.
    models = client.get("/api/models").json()["models"]
    assert [m["id"] for m in models] == ids

    # --- Check 3: Load one real image from the test dataset ---
    # We find the first 'Forest' test image to send in our test prediction requests.
    test = pd.read_csv(SPLITS_CSV).query("split == 'test'")
    row = test[test["class_name"] == "Forest"].iloc[0]
    image_bytes = (DATA_DIR / row["path"]).read_bytes()

    # --- Check 4: Test prediction for every available model ---
    for model_id in ids:
        # Simulate an HTTP POST uploading the image with the chosen model ID
        answer = client.post(
            "/api/predict",
            files={"image": ("tile.jpg", image_bytes, "image/jpeg")},
            data={"model": model_id}
        ).json()

        # Verify the predicted label is one of the 10 known EuroSAT classes
        assert answer["label"] in CLASSES and 0 <= answer["confidence"] <= 1, answer["label"]

        # Verify top-3 predictions structure and consistency
        assert [t["label"] for t in answer["top3"]][0] == answer["label"] and len(answer["top3"]) == 3
        assert abs(answer["confidence"] - answer["top3"][0]["prob"]) < 1e-9

        # Verify Grad-CAM overlay is returned as a base64 encoded PNG image
        assert answer["gradcam"].startswith("data:image/png;base64,")
        overlay = Image.open(io.BytesIO(__import__("base64").b64decode(answer["gradcam"].split(",", 1)[1])))
        assert overlay.mode == "RGBA" and overlay.size == (64, 64), (overlay.mode, overlay.size)

        print(f"predict OK with {model_id}: {answer['label']} ({answer['confidence']:.1%}), true class Forest")

    # --- Check 5: Verify input validation and error handling ---
    # Sending a plain text file instead of an image must return HTTP 400 Bad Request
    assert client.post("/api/predict", files={"image": ("x.txt", b"not an image", "text/plain")}).status_code == 400

    # Requesting a non-existent model name must also return HTTP 400 Bad Request
    assert client.post("/api/predict", files={"image": ("t.jpg", image_bytes, "image/jpeg")}, data={"model": "nope"}).status_code == 400

    # --- Check 6: Verify the metrics endpoint ---
    metrics = client.get("/api/metrics").json()
    # Must contain a 10x10 confusion matrix
    assert len(metrics["confusion"]) == 10 and all(len(r) == 10 for r in metrics["confusion"])
    # Total count in confusion matrix must equal our 4050 test set size, with accuracy > 90%
    assert sum(map(sum, metrics["confusion"])) == 4050 and metrics["accuracy"] > 90
    # Data efficiency curves must have points for all 5 tested training fractions (5%, 10%, 25%, 50%, 100%)
    assert len(metrics["efficiency"]["scratch"]) == len(metrics["efficiency"]["transfer"]) == 5
    print(f"metrics OK: accuracy {metrics['accuracy']}%, macro-F1 {metrics['macro_f1']}%")

    # --- Check 7: Verify frontend web pages and redirects ---
    # Root /app.html without ?api query should redirect to attach the API URL parameter
    assert client.get("/app.html", follow_redirects=False).status_code in (302, 307)
    # /app.html with ?api specified should return status 200 OK
    assert client.get("/app.html?api=http://testserver").status_code == 200
    # Landing page and hero graphic asset should load with status 200 OK
    assert client.get("/Eyes_in_the_Sky_landing.html").status_code == 200
    assert client.get("/Moonlit%20Satellite%20and%20Rocket%20Trail.png").status_code == 200

    print("pages OK")
    print("test_server: OK")


if __name__ == "__main__":
    main()
