# Eyes in the Sky: Land-Use Classification from Satellite Images (Group 5)

## Overview

Atria University, "Neural Network and CNN Sprint" - Group 5 capstone.

We classify EuroSAT satellite images (RGB, 64x64 pixels, 27,000 images, 10 land-use classes) with a CNN built in PyTorch, and we compare:

- a CNN we design ourselves, with ablations (one change at a time),
- transfer learning (ResNet-18 and MobileNetV2, frozen vs fully fine-tuned),
- how accuracy changes when we use less training data (data-efficiency study),
- what the model looks at and where it fails (confusion matrix, Grad-CAM, t-SNE).

A small website (FastAPI + plain HTML/CSS/JS in `app/`, built with AI help and disclosed in the report) will show predictions live in the demo. It runs fully offline.

Team rules:

- Never commit the dataset or model weights (`.gitignore` blocks them).
- Every member commits from their own GitHub account, on a feature branch, and opens a pull request.
- All results go in `results/metrics.csv` (columns: `model,fraction,seed,accuracy,macro_f1`).
- Every file has an owner who can explain every line of it.

Project state: this is the empty skeleton. Only `check_setup.py` is real code today.

## Setup (Windows)

Use PowerShell. Do the steps in order.

1. Install **Python 3.12** from python.org (tick **"Add python.exe to PATH"**) and **Git** from git-scm.com. Check both work:

   ```
   py -3.12 --version
   git --version
   ```

2. Clone the repo and go into the folder:

   ```
   git clone https://github.com/Nitinshakthi7/Eyes-in-the-Sky.git
   cd Eyes-in-the-Sky
   ```

3. Make a virtual environment (a private box for this project's Python packages) and switch it on:

   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   You should now see `(.venv)` at the start of the line. If PowerShell says "running scripts is disabled on this system", run this once, close and reopen PowerShell, and repeat step 3 from the `Activate.ps1` line:

   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. Install PyTorch. Everyone installs the **same version** (2.14.1 / 0.29.1). Do **not** use plain `pip install torch`: on Windows it can give you a CPU-only build. First run `nvidia-smi`, then pick ONE line:

   - NVIDIA GPU and "Driver Version" is **580 or higher** (this is the GPU command):

     ```
     pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
     ```

   - NVIDIA GPU but the driver is **lower than 580** (or update your NVIDIA driver instead):

     ```
     pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
     ```

   - No NVIDIA GPU (`nvidia-smi` says "not recognized"), so the CPU is used:

     ```
     pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
     ```

5. Install the other packages:

   ```
   pip install -r requirements.txt
   ```

6. Check your laptop and download the data and pretrained weights (takes a few minutes; the download is about 150 MB). Use your first name:

   ```
   python check_setup.py --name YourName --download
   ```

   Every line says OK, WARN or FAIL with a one-line fix. Fix any FAIL and run it again. At the end it prints one line for the team chat: paste it there.

7. Send your report to the team through a pull request (replace `yourname` with your first name in lowercase):

   ```
   git checkout -b setup/yourname
   git add docs/setup/yourname.md
   git commit -m "Add setup report for YourName"
   git push -u origin setup/yourname
   ```

   Then open the pull request on GitHub (the page shows a "Compare & pull request" button).

## How to run

Right now only the setup check works (run it inside the activated venv, from the repo root):

```
python check_setup.py --name YourName
```

The other steps are filled in as each part is finished (TODO):

| Step | Who | How to run |
|---|---|---|
| Split the data (fixed 70/15/15) and EDA | A | TODO (`notebooks/01_eda.ipynb`, `src/dataset.py`) |
| Scratch CNN and ablations | B | TODO (`notebooks/02_baseline_cnn.ipynb`) |
| Transfer learning | C | TODO (`notebooks/03_transfer_learning.ipynb`) |
| Train one model: `train(model_name, fraction, seed)` | D | TODO (`src/train.py`) |
| Data-efficiency study | D | TODO (`notebooks/05_data_efficiency.ipynb`) |
| Evaluation, Grad-CAM, t-SNE | E | TODO (`notebooks/04_evaluation_and_explainability.ipynb`) |
| Website (offline) | - | TODO (`app/`) |

## Results

TODO (filled in when the experiments are done; everything comes from `results/metrics.csv` and `results/figures/`).

## Team workflow (relay)

**Our full step-by-step plan with one task card per person: [docs/plan.md](docs/plan.md).** Your own daily log and line-by-line code notes: `docs/log/<yourname>.md` (the file already exists; fill it in every day you work).

We work like a relay race: each person pulls the previous person's work, reads it, adds their part and pushes. The person who handed over the baton reviews and merges the pull request. Nobody merges their own work.

**Lap 1 (Mon-Tue): a small version that works from start to end**

1. Nitin (D): skeleton, `utils.py`, invites everyone
2. Shrivalli (A): `dataset.py`, fixed split `data/splits.csv`, EDA notebook
3. Tejash (B): scratch CNN and a first baseline
4. Rachana (C): ResNet-18 and MobileNetV2, one first run each
5. Nitin (D): `train(model_name, fraction, seed)` and the first data-efficiency runs
6. Rohan (E): confusion matrix, per-class scores, Grad-CAM, t-SNE

**Lap 2 (Wed-Thu):** everyone pulls again and goes deep on their own part (ablations, full runs, analysis, report). Gate 2 is Wednesday 21:00: results frozen. Thursday: report, README, mock viva. Friday: demo.

**Every hand-off:**

1. `git checkout main`, then `git pull`
2. `git checkout -b <part>-<what>-<name>` (for example `b-scratch-cnn-tejash`)
3. Do your work, then `git add <your files>`, `git commit -m "..."`, `git push -u origin <your branch>`
4. Open a pull request. Write 3 lines: what you did, what the next person must read first, what they do next. Also write 3 lines on what you understood from the previous part.
5. The previous person reviews and merges it. The next person starts again at step 1.

**While you wait for the baton:** read the previous two parts, write your 6 viva questions in `docs/viva.md`, draft your report section, and test your code on dummy data. If the baton is more than 1 hour late, start from stubs; the late person still finishes their part and keeps the credit.

## Contribution table

| Member | GitHub username | Part | Key files |
|---|---|---|---|
| Srivalli | | A - Data + EDA + fixed split | `src/dataset.py`, `notebooks/01_eda.ipynb`, `data/splits.csv` |
| Tejash | | B - Scratch CNN + ablations | `src/models.py` (scratch CNN), `notebooks/02_baseline_cnn.ipynb` |
| Rachana | | C - Transfer learning (ResNet-18, MobileNetV2) | `src/models.py` (transfer models), `notebooks/03_transfer_learning.ipynb` |
| Nitin (team leader) | | D - Shared `train()` + data-efficiency study + repo | `src/train.py`, `src/utils.py`, `notebooks/05_data_efficiency.ipynb`, `check_setup.py`, `results/metrics.csv` |
| Rohan | | E - Evaluation + Grad-CAM + t-SNE | `src/evaluate.py`, `src/gradcam.py`, `notebooks/04_evaluation_and_explainability.ipynb` |

The website in `app/` is built by AI after Gate 2 (Wednesday night) and disclosed in the report. No member owns it.

## AI usage

We write down every use of AI help in [docs/ai_usage.md](docs/ai_usage.md) and describe it in the report.
