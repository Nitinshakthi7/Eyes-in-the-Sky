"""test_server.py - a small check of the website backend (no pytest needed).

Run it from the Backend folder:  python -m app.test_server
Needs the weights in models/ and the dataset in data/ (it sends one real test image).
"""

import io

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from app.server import CLASSES, app, available
from src.utils import DATA_DIR, SPLITS_CSV

client = TestClient(app)


def main():
    ids = available()
    assert ids, "put at least one <model>_frac1_seed0.pth in models/"

    models = client.get("/api/models").json()["models"]
    assert [m["id"] for m in models] == ids

    test = pd.read_csv(SPLITS_CSV).query("split == 'test'")
    row = test[test["class_name"] == "Forest"].iloc[0]
    image_bytes = (DATA_DIR / row["path"]).read_bytes()
    for model_id in ids:
        answer = client.post("/api/predict", files={"image": ("tile.jpg", image_bytes, "image/jpeg")}, data={"model": model_id}).json()
        assert answer["label"] in CLASSES and 0 <= answer["confidence"] <= 1, answer["label"]
        assert [t["label"] for t in answer["top3"]][0] == answer["label"] and len(answer["top3"]) == 3
        assert abs(answer["confidence"] - answer["top3"][0]["prob"]) < 1e-9
        assert answer["gradcam"].startswith("data:image/png;base64,")
        overlay = Image.open(io.BytesIO(__import__("base64").b64decode(answer["gradcam"].split(",", 1)[1])))
        assert overlay.mode == "RGBA" and overlay.size == (64, 64), (overlay.mode, overlay.size)
        print(f"predict OK with {model_id}: {answer['label']} ({answer['confidence']:.1%}), true class Forest")

    assert client.post("/api/predict", files={"image": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    assert client.post("/api/predict", files={"image": ("t.jpg", image_bytes, "image/jpeg")}, data={"model": "nope"}).status_code == 400

    metrics = client.get("/api/metrics").json()
    assert len(metrics["confusion"]) == 10 and all(len(r) == 10 for r in metrics["confusion"])
    assert sum(map(sum, metrics["confusion"])) == 4050 and metrics["accuracy"] > 90
    assert len(metrics["efficiency"]["scratch"]) == len(metrics["efficiency"]["transfer"]) == 5
    print(f"metrics OK: accuracy {metrics['accuracy']}%, macro-F1 {metrics['macro_f1']}%")

    assert client.get("/app.html", follow_redirects=False).status_code in (302, 307)
    assert client.get("/app.html?api=http://testserver").status_code == 200
    assert client.get("/Eyes_in_the_Sky_landing.html").status_code == 200
    assert client.get("/Moonlit%20Satellite%20and%20Rocket%20Trail.png").status_code == 200
    print("pages OK")
    print("test_server: OK")


if __name__ == "__main__":
    main()
