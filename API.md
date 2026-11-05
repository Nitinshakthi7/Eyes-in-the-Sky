# API contract for the UI (`app.html`)
Open the UI with `app.html?api=http://localhost:8000`. Without `?api=` it runs in demo mode with simulated results.
Enable CORS on the server so the page can call it.

**GET /api/models** → `{"models":[{"id":"resnet18","label":"ResNet-18 (fine-tuned)"}]}`

**POST /api/predict** (multipart: `image` = file, `model` = id) →
`{"label":"Forest","confidence":0.97,"top3":[{"label":"Forest","prob":0.97},{"label":"Pasture","prob":0.02},{"label":"HerbaceousVegetation","prob":0.01}],"gradcam":"data:image/png;base64,..."}`
`gradcam` is a transparent PNG heat-map the same aspect as the input (the UI overlays it). Class names must match EuroSAT folder names.

**GET /api/metrics** →
`{"accuracy":94.9,"macro_f1":94.8,"confusion":[[10 rows x 10 ints, class order as EuroSAT]],"efficiency":{"fractions":[5,10,25,50,100],"scratch":[..5 accuracies in %..],"transfer":[..]}}`
