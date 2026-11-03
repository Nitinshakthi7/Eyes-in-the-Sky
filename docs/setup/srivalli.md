# Setup report: Srivalli

| Check | Status | Detail |
|---|---|---|
| Python | WARN | 3.13.14 - team standard is 3.12 |
| torch | OK | 2.14.1+cpu |
| torchvision | OK | 0.29.1+cpu |
| numpy | OK | 2.5.4 |
| pandas | OK | 3.1.0 |
| matplotlib | OK | 3.11.2 |
| scikit-learn | OK | 1.9.1 |
| pillow | OK | 13.0.0 |
| fastapi | OK | 0.162.0 |
| uvicorn | OK | 0.54.0 |
| python-multipart | OK | 0.0.33 |
| notebook | OK | 7.6.4 |
| ipykernel | OK | 7.4.0 |
| Git | OK | git version 2.51.0.windows.1 |
| Git user.name | OK | Srivalli |
| Git user.email | OK | srivallikarthikaa.dp@atriauniversity.edu.in |
| NVIDIA GPU | WARN | no NVIDIA GPU or driver found |
| PyTorch | OK | version 2.14.1+cpu, CUDA build none (CPU-only build), cuda available = False |
| PyTorch GPU | WARN | no GPU for PyTorch, will use CPU |
| Disk space | OK | 287.1 GB free |
| Speed Small CNN | OK | 97 images/s, 195 s per epoch, 65.1 min for 20 epochs |
| Speed ResNet-18 | OK | 46 images/s, 413 s per epoch, 137.6 min for 20 epochs |
| Verdict | OK | Take a non-training part (A or E), or train on a teammate's GPU |
| EuroSAT dataset | OK | 27000 images, 10 classes |
| ResNet-18 weights | OK | downloaded to the torch cache |
| MobileNetV2 weights | OK | downloaded to the torch cache |

Team line:

    Srivalli | Python 3.13.14 | torch 2.14.1+cpu + CUDA none (CPU-only build) | CPU only | ResNet-18 413 s/epoch | Take a non-training part (A or E), or train on a teammate's GPU
