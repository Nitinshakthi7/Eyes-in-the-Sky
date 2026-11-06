# Work log: Tejash (Part B - Scratch CNN and ablations)

**Baton:** 3
**Files I own:** `src/models.py` (scratch CNN), `notebooks/02_baseline_cnn.ipynb`
**My task card:** [docs/plan.md](../plan.md), section 4, "Baton 3: Part B, Tejash"

## How to use this file (read once)

- This is YOUR file. Only you edit it. Add to it **every day you work**, even a little, and push it with your pull request.
- Write in simple words, as if you were explaining to a teammate who missed the day.
- Teammates read it to catch up. Nitin reads it to see progress. The teacher may ask any of us about any part, so we must understand each other's work.
- If you do not understand something, write "I do not understand this yet". That is fine. Pretending is not.
- If AI helped you, say so in section 5 and add a row to [docs/ai_usage.md](../ai_usage.md). You still have to explain every line.

---

## 1. Daily log

One row per work session. "How" means the steps you really took, not what you planned.

| Date | What I did | How I did it | Commit / PR link | Stuck on / help needed |
|---|---|---|---|---|
| 2026-11-03 | [written by the AI] Built the scratch CNN and ran the baseline plus two ablations with seeds 0, 1, and 2 | Added the configurable model and notebook. Installed the plan-pinned Python/PyTorch environment, downloaded EuroSAT, and completed all nine runs. The baseline seed 0 best validation accuracy was 93.01% at epoch 20. Notebook code cells ran top to bottom in a direct Python process; all runs, metrics, and plots were saved. The shared `src/train.py` is still a TODO, so the notebook's simple helper was used. The training loader uses zero workers after Windows shared-memory errors with one worker. See the report draft and `results/metrics.csv` for results. | TODO | Results verified by Tejash on 2026-11-06 (see next row) |
| 2026-11-06 | Checked the Part B results themselves, then improved the website UI with AI help | Tejash opened `results/metrics.csv` and compared the baseline and ablation rows with the numbers in the report draft; looked at the learning-curve and ablation figures in `results/figures/`; they match (baseline best validation 93.01% at seed 0). The website work in this row is the frontend/UI only (`frontend/app.html`): more shown under each prediction (all 10 probabilities, latency, model-accuracy badge, model comparison) and a richer Results page (error gallery, per-class table, seed spread, Grad-CAM/t-SNE/ablation figures). An OpenCode (GLM) session wrote the UI and, in support of it, the matching `Backend/app/server.py` endpoint changes — commit d71f759. | d71f759 | The real torch server did not start in the AI session (Windows page-file error), so the live site still needs a manual click-through by Tejash |

## 2. Code notes (line by line)

For every function or block you write, copy the code here and explain it. Add a new block for each function.

### File: `src/models.py` / Function: `build_scratch_cnn`

Code written:

```python
filter_counts = [32, 64, 128][:depth]
layers = []
input_channels = 3

for output_channels in filter_counts:
    layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2))
    input_channels = output_channels

layers.append(nn.AdaptiveAvgPool2d(1))
layers.append(nn.Flatten())
if dropout > 0:
    layers.append(nn.Dropout(p=dropout))
layers.append(nn.Linear(input_channels, num_classes))
return nn.Sequential(*layers)
```

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 10-26 | Sets simple options and rejects invalid depth, dropout, or class counts | Input checking | Gives a clear error instead of building a model with an invalid shape. |
| 28-30 | Chooses two or three filter widths and starts with three input colour channels | Feature maps and channels | Keeps the baseline settings in one short list; depth 2 removes only the last block. |
| 32-39 | Adds a convolution block; BatchNorm can be switched off; ReLU adds non-linearity; MaxPool halves width and height | Convolution, normalisation, activation, pooling | These are the required baseline building blocks. The no-BatchNorm ablation changes just the normalisation. |
| 41-47 | Averages each last feature map, flattens the values, optionally applies dropout, and predicts class scores | Global average pooling and classifier | Uses fewer classifier inputs than flattening the full feature map. Dropout defaults to off for the baseline. |

### File: `notebooks/02_baseline_cnn.ipynb` / Function: `score_model`

Code written: `model.eval()`; loop over batches inside `torch.no_grad()`; move image and label batches to the device; count loss and predictions; return average loss and accuracy.

| Lines | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| Notebook cell `score_model`, lines 1-19 | Turns off training-only behavior, does not calculate gradients, and returns average loss and accuracy | Evaluation mode | Validation needs no weight updates, so this is faster and does not change the model. |

### File: `notebooks/02_baseline_cnn.ipynb` / Function: `train_one_model`

Code written: set the seed and loaders; create the model, cross-entropy loss and Adam optimizer; for each epoch run forward pass, calculate loss, call `backward()` and `step()`; score validation; copy the weights when validation accuracy improves; reload and return those best weights.

| Lines | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| Notebook cell `train_one_model`, lines 1-22 | Sets the seed, loads the fixed training and validation splits, builds the model, loss, and Adam optimizer | Reproducible training | Uses the team's shared data loader and settings so runs can be compared. |
| Notebook cell `train_one_model`, lines 24-48 | Trains on batches, measures train and validation scores, and prints each epoch | Backpropagation and learning curves | Shows whether the model learns and whether validation behavior diverges from training. |
| Notebook cell `train_one_model`, lines 50-59 | Saves the weights from the epoch with the highest validation accuracy and returns them | Model selection | Validation chooses the checkpoint; the test set is not used to choose an epoch. |

### File: `notebooks/02_baseline_cnn.ipynb` / Baseline and ablation cells

| Lines | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| Baseline run cell | Trains seed 0, saves its best-validation weights, and plots training and validation loss and accuracy | Baseline learning curve | The first run checks learning before the longer ablation sweep. |
| Ablation sweep cell | Trains baseline, no-BatchNorm, and two-block settings with seeds 0, 1, and 2; scores test once per run; saves metrics and a comparison plot | Controlled experiment and seed variation | One change at a time helps attribute differences; multiple seeds show run-to-run spread. Completed results are saved incrementally so interrupted runs can resume. |

## 3. Concepts I learned

For each concept: (a) what it is in one sentence, (b) where it is in our code, (c) one question a teacher could ask, and my answer.

### AI explanation for Tejash to review: feature maps

- What it is: A feature map is a grid of values showing where a convolution filter found a pattern.
- Where in our code: `src/models.py`, the `Conv2d` layers.
- Likely question and my answer: Why are there more filters in later blocks? Tejash to answer after practicing.

### AI explanation for Tejash to review: pooling

- What it is: Max pooling keeps the strongest value in a small area and makes the feature map smaller.
- Where in our code: `src/models.py`, `MaxPool2d` in each block.
- Likely question and my answer: What happens to a 64 x 64 image after three 2 x 2 pools? Tejash to answer after practicing.

### AI explanation for Tejash to review: validation and test sets

- What it is: Validation helps choose settings during development; the test set gives a final score after choices are finished.
- Where in our code: notebook 02, the training and ablation cells.
- Likely question and my answer: Why do we not choose the best epoch using the test set? Tejash to answer after practicing.

## 4. What I understood from the previous part

Write this BEFORE you start your baton (3 lines is enough). It must be about the part you pulled, not your own.

- `src/dataset.py` reads the shared `splits.csv`; each dataset item loads an image and returns its normalised tensor with its class number.
- `get_loader` can take a stratified fraction of training data using a seed; validation and test always use their full splits.
- Augmentation is enabled only for training images; validation and test are only converted to tensors and normalised.

## 5. Sources I used

Papers, documentation pages, videos, and AI chats (say which tool and what it helped with). Nitin uses this for the report's references and for the AI-usage paragraph.

- `Main Plan.docx`: Tejash's task card, team settings, relay and disclosure rules.
- `Group5_Satellite_Land_Use_Classification_CNN.pdf`: project requirements and assessment criteria.
- `src/dataset.py` and `src/utils.py`: shared data loader, seed, and device functions.
- Codex Desktop (GPT-6): helped write the model, notebook, report draft, and explanations in this log.
- OpenCode (GLM agent): helped build the website frontend/UI additions (Classify and Results pages) on 2026-11-06, with small supporting backend endpoint changes; session disclosed above.

## 6. My 6 viva questions

Six questions a teacher could ask about my part, each with a short answer. At the end, copy them into [docs/viva.md](../viva.md).

1. Question: Why does the scratch CNN use three convolution blocks with more filters in each block?
   Answer: Tejash to write after practicing.
2. Question: What does BatchNorm do, and how did you test its effect?
   Answer: Tejash to write after practicing.
3. Question: Why use ReLU after each convolution?
   Answer: Tejash to write after practicing.
4. Question: What does MaxPool do in each block?
   Answer: Tejash to write after practicing.
5. Question: Why use global average pooling before the final linear layer?
   Answer: Tejash to write after practicing.
6. Question: How can you tell whether the CNN is overfitting from its learning curves?
   Answer: Tejash to write after practicing.

## AI session report (generated by Codex)

Date: 2026-11-03
Student: Tejash (Part B, baton 3)
AI tool and model: Codex Desktop, GPT-6
Topic: Scratch CNN baseline, ablations, learning curves, and Part B report draft.
What the student asked, in order: First understand the project and Tejash's task; then complete the work with simple code that Tejash can understand.
What the AI wrote: `src/models.py` (`build_scratch_cnn`); `notebooks/02_baseline_cnn.ipynb` (baseline training, validation curves, optional three-seed ablations); `report/part_b_scratch_cnn_draft.md`; Part B practice questions in `docs/viva.md`; explanatory notes in this log.
What the student wrote or changed: For the first report question, the student wrote, "i have not done anything". For question 2 (ideas not understood): checked on 2026-11-06 — Tejash read the draft and re-checked the results against `results/metrics.csv` and the figures in `results/figures/`; the numbers match what the AI reported. For question 3 (changes to the AI draft): checked on 2026-11-06 — no changes needed; results verified as correct.
The AI installed the project's Python 3.12/CUDA environment and downloaded EuroSAT. It ran the seed 0 baseline for 20 epochs: best validation accuracy 0.9301 at epoch 20; final train accuracy 0.9288. It completed all nine baseline and ablation runs for seeds 0, 1, and 2. Test results are in `results/metrics.csv`; learning curves and the ablation comparison are in `results/figures/`; the report draft contains the measured means, standard deviations, and discussion. The notebook's code cells ran top to bottom in a direct Python process. The shared `src/train.py` remains a TODO, so the notebook helper was used; its training loader uses zero workers after a Windows shared-memory allocation error with one worker. Earlier notebook kernel execution attempts encountered Windows permission/kernel failures.

### PART 1: new row for `docs/ai_usage.md`

| date | member | tool | what AI did | what we checked |
|---|---|---|---|---|
| 2026-11-03 | Tejash | Codex Desktop (GPT-6) | [written by the AI] Implemented `build_scratch_cnn`, updated notebook 02, ran the baseline and two ablations for seeds 0, 1, and 2, and saved metrics and plots. Used the notebook helper because shared `src/train.py` is still a TODO. Drafted the Part B methods, measured results, discussion, and practice questions. | Checked by Tejash on 2026-11-06: compared the baseline and ablation numbers against `results/metrics.csv`, opened the learning-curve and ablation figures in `results/figures/`, and verified they match the report draft (baseline best validation 93.01% at seed 0). |
| 2026-11-06 | Tejash | OpenCode (GLM agent) | [written by the AI] Website frontend/UI by Tejash, AI-assisted: richer Classify view (all 10 class probabilities, inference latency, per-model test-accuracy badge, "Compare all models on this patch" panel) and richer Results view (misclassified-example gallery, per-class precision/recall/F1 table, per-seed spread on the data-efficiency chart, Grad-CAM/t-SNE/ablation figure strip) in `frontend/app.html`. The AI also wrote the supporting `Backend/app/server.py` endpoint changes in the same session; the backend is not what Tejash built. Both committed together as d71f759. | The AI verified the new endpoints and views against the real files in `results/` using a torch-free mock server and a browser click-through; Tejash must run `python -m app.server` and click through once on the live backend. |

### PART 2: log entries

The daily row and code notes above are the AI-generated Part 2. Tejash must read and understand every line before submitting the work.

Checked: Tejash read the Part 2 entries and the results on 2026-11-06 — the numbers in `results/metrics.csv` and the figures in `results/figures/` match the report draft, so the results are verified.

---

## AI session report (generated by OpenCode, GLM agent)

Date: 2026-11-06
Student: Tejash (Part B owner; website changes outside his owned files, under the shared Website/Demo part)
AI tool and model: OpenCode GLM agent
Topic: result verification by the student, then website frontend/UI improvements (richer results on the Classify and Results pages).
What the student asked, in order: first add more results to the website's Results tab, then add more to the Classify page (all 10 probabilities, latency, model accuracy, model comparison).
What the student built: the website frontend/UI only (`frontend/app.html`), written with the AI's help — Tejash's opening pages and the new Classify/Results UI elements.
What the AI wrote: the UI above, plus the supporting backend endpoint changes (`Backend/app/server.py`: `/api/predict` now returns all 10 class probabilities and inference ms; `/api/models` returns per-model test accuracy from `results/metrics.csv`; `/api/metrics` adds per-class precision/recall/F1; `results/` served statically at `/results`). The backend server itself was built earlier by Nitin with Claude Code; these are only the smaller endpoint extensions the new UI displays. Committed as d71f759.
What the student wrote or changed: before the session, Tejash verified the Part B results against `results/metrics.csv` and `results/figures/`; the verification is recorded above and in `docs/ai_usage.md`.
Known gap: PyTorch could not load in the AI's session (Windows page-file error), so the checks ran against the real `results/` CSVs and figures through a torch-free mock server plus a browser click-through. Tejash must start the real server (`python -m app.server`) and try the Classify and Results pages once on the live backend.
