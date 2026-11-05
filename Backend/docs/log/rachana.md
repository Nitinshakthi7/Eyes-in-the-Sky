# Work log: Rachana (Part C - Transfer learning (ResNet-18, MobileNetV2))

**Baton:** 4
**Files I own:** `src/models.py` (transfer models), `notebooks/03_transfer_learning.ipynb`, `report/part_c_transfer_learning_draft.md`
**My task card:** [docs/plan.md](../plan.md), section 4, "Baton 4: Part C, Rachana"

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
| 2026-11-02 | Prepared laptop and environment for transfer-learning task: installed and verified Python, Git, PyTorch, torchvision and project dependencies; verified PyTorch detects NVIDIA RTX 4050 GPU via CUDA; downloaded and verified EuroSAT dataset, ResNet-18 weights and MobileNetV2 weights. | Guided by ChatGPT for setup commands, which I copied, pasted and ran in the terminal myself, sharing outputs for validation. Ran `python check_setup.py --name Rachana --download`. Configured Git global identity, created branch `setup/rachana`, regenerated setup report and staged it using Antigravity. | (merged in Gate 1) | Still learning Git and its commands. Understand that CUDA allows PyTorch to use GPU to accelerate training, but not yet completely clear on why CUDA is specifically needed and how it works internally. |
| 2026-11-03 | Pulled `main` after Batons 2, 3 and 5 (`8ecbc56`), created branch `c-transfer-rachana`, reviewed Part B (`src/models.py`, `notebooks/02_baseline_cnn.ipynb`) and Nitin's shared `src/train.py`. Implemented `build_transfer_model(name, freeze)` in `src/models.py` for ResNet-18 and MobileNetV2, verified parameter counts, and ran the four Lap 1 seed-0 experiments (`resnet18_frozen`, `resnet18_full`, `mobilenet_v2_frozen`, `mobilenet_v2_full`) in `notebooks/03_transfer_learning.ipynb`. Saved seed-0 learning curves and `transfer_seed0_histories.json`. | Used ChatGPT to break the Baton 4 task card into smaller steps and Gemini in Antigravity to help write `build_transfer_model` and set up `notebooks/03_transfer_learning.ipynb`. Checked Python/CUDA (`Python 3.12.10`, `PyTorch 2.14.1+cu130`, `RTX 4050`) after finding multiple Python installations on my laptop. Ran the four seed-0 configurations in the native Jupyter kernel and checked the validation and test scores. | Branch `c-transfer-rachana` (pending PR) | Needed clarification on how `model.fc` in ResNet-18 and `model.classifier[1]` in MobileNetV2 are replaced for 10 classes, and why full fine-tuning uses a 10x smaller learning rate (`0.0001` vs `0.001`). |
| 2026-11-04 | Completed Lap 2 multi-seed experiments (seeds 0, 1, 2 across all 4 transfer configurations = 12 runs total). Handled a CUDA display driver reset during the second batch, recovered the seven completed Lap 2 training logs into `results/figures/transfer_all_seeds_histories.json`, and ran the final `mobilenet_v2_full` (seed 2) experiment in Jupyter using skip logic. Updated the 3-seed summary table to use sample standard deviation (`ddof=1`), generated `transfer_vs_scratch_comparison.png`, drafted `report/part_c_transfer_learning_draft.md`, and added 3-layer artifact validation guards in `notebooks/03_transfer_learning.ipynb`. | Used Gemini in Antigravity and ChatGPT iteratively. After the CUDA interruption, I restarted the Jupyter kernel, reran the setup cells, and ran cell 8 so the skip check skipped the 11 finished runs and trained only `mobilenet_v2_full` seed 2. I checked `results/metrics.csv` (9 Part B rows + 12 Part C rows = 21 rows) and all 12 `.pth` checkpoints in `models/`. I asked for fixes to `ddof=1`, non-causal wording for the frozen-vs-full comparison (since learning rate also changes), and an explanation of BatchNorm running statistics when `freeze=True`. | Branch `c-transfer-rachana` (pending PR) | I understand the main transfer-learning workflow now, though I still want to strengthen my understanding of BatchNorm running statistics (`running_mean` and `running_var`), CPU vs GPU random augmentation reproducibility, and deeper differences between frozen feature extraction and full fine-tuning. |

---

## 2. Code notes (line by line)

### File: `src/models.py` / Function: `build_transfer_model(name: str, freeze: bool = True)`

```python
def build_transfer_model(name: str, freeze: bool = True):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported transfer model {name!r}; expected 'resnet18' or 'mobilenet_v2'")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif name == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    return model
```

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 52 | Defines `build_transfer_model(name, freeze=True)` which takes the backbone name and whether to freeze the pretrained layers. | Function definition / modular model builder | Matches `src/train.py`, where `build_model()` calls `models.build_transfer_model(backbone, freeze)` for all four Part C model names. |
| 59-62 | Loads either `resnet18` or `mobilenet_v2` from `torchvision.models` with `DEFAULT` pretrained ImageNet-1K weights from the local cache. | Pretrained weights / Transfer learning | Gives the model starting filters learned on 1.28M ImageNet photos instead of starting from random numbers. |
| 63-64 | Raises a `ValueError` if an unknown model name is passed. | Input validation | Fails early with a clear message if someone types the wrong model name. |
| 66-69 | If `freeze` is `True`, loops over every parameter in the pretrained model and sets `param.requires_grad = False`. | Freezing backbone parameters | Tells PyTorch not to compute gradients or update weights for the pretrained backbone layers during backpropagation. |
| 73-74 | For ResNet-18, replaces `model.fc` (which outputs 1,000 ImageNet classes from 512 features) with a new `nn.Linear(512, 10)` layer (`5,130` trainable parameters). | Classification head replacement | EuroSAT has 10 land-use classes, not 1,000. Newly created `nn.Linear` layers have `requires_grad = True` by default, so the new head always learns even when the backbone is frozen. |
| 75-76 | For MobileNetV2, replaces `model.classifier[1]` (which takes 1,280 features) with `nn.Linear(1280, 10)` (`12,810` trainable parameters) while keeping `model.classifier[0]` (`Dropout(p=0.2)`). | Classification head replacement | Adapts MobileNetV2 to our 10 classes. Total parameters are `2,236,682` (about 5x smaller than ResNet-18's `11,181,642`). |
| 78 | Returns the ready PyTorch model. | Model output | Returned to `build_model()` in `src/train.py`, which pairs it with `lr = 0.001` when `freeze=True` or `lr = 0.0001` when `freeze=False`. |

---

### File: `notebooks/03_transfer_learning.ipynb` / Key Notebook Cells

| Cell ID | What it does (simple words) | Concept | Why we used it |
|---|---|---|---|
| `cell-01-imports` & `cell-02-config` | Sets the working directory to the repo root, imports `train`, `build_model` and paths from `src/`, checks the GPU (`RTX 4050`), and builds a table showing `freeze_backbone`, `learning_rate`, `trainable_params` and `total_params` for all four configurations. | Configuration & parameter verification | Before running training, I wanted to verify that `resnet18_frozen` only trains `5,130` parameters, `mobilenet_v2_frozen` trains `12,810`, and the `full` versions train all parameters (`11,181,642` and `2,236,682`) at `lr = 0.0001`. |
| `cell-04-run-seed0` | Checks three artifact layers (`results/metrics.csv`, `models/*_frac1_seed0.pth`, and `results/figures/transfer_seed0_histories.json`) for each seed-0 model. If all three are valid, it skips training and loads `seed0_runs` from JSON; if partial artifacts exist, it raises a `RuntimeError`; if none exist, it runs `train()` for 20 epochs. | Lap 1 seed-0 execution & 3-layer rerun guard | Prevents accidental retraining or overwriting of completed seed-0 results when rerunning the notebook, while still loading `seed0_runs` so the plotting cell works. |
| `cell-06-summary-plots` | Builds a summary table of the four seed-0 runs (`best_epoch`, `val_accuracy`, `test_accuracy`, `test_macro_f1`, `seconds_per_epoch`) and plots 20-epoch training and validation loss and accuracy curves, saving `results/figures/transfer_learning_curves_seed0.png`. | Learning-curve visualisation | Lets us see how fast each configuration converges and whether validation loss diverges from training loss (overfitting). |
| `cell-08-run-lap2` | Loops over all 4 models and `SEEDS = [0, 1, 2]`. Validates that each run has 1 valid row in `results/metrics.csv`, a non-empty `.pth` checkpoint in `models/`, and a complete 20-epoch entry in `transfer_all_seeds_histories.json`. Skips verified runs, raises `RuntimeError` if artifacts are inconsistent, and calls `train()` only for genuinely missing runs. | Lap 2 multi-seed execution & artifact validation | Allowed me to resume cleanly after the CUDA interruption on seed 2 and ensures nobody accidentally overwrites the 12 finished runs or checkpoints. |
| `cell-08b-lap2-recovery-table` | Loads `results/figures/transfer_all_seeds_histories.json`, checks that all 12 runs have 20 epochs, cross-checks every test accuracy and macro-F1 against `results/metrics.csv`, and displays a 12-row provenance table showing which runs have exact `float64` histories (seed 0) vs rounded stdout log histories (seeds 1 and 2). | Experiment audit & log recovery | Keeps the notebook honest about where the seed-1 and seed-2 epoch curves came from (`task-594.log` and `cell-08` stdout) without inventing extra decimal places. |
| `cell-10-lap2-summary-table` | Reads `results/metrics.csv` for `scratch_cnn` and the four transfer models (`seeds 0, 1, 2`, `fraction = 1.0`), calculates the 3-seed mean and sample standard deviation (`std(ddof=1)`), and computes the percentage-point difference against `scratch_cnn`. | Multi-seed statistical summary (`ddof=1`) | Using `ddof=1` matches Tejash's Part B baseline calculation (`92.72% ± 0.45%` accuracy, `92.48% ± 0.51%` macro-F1) so all five models are compared on the exact same statistical formula. |
| `cell-12-comparison-plot` | Reads the 15 rows (`5 models × 3 seeds`) from `results/metrics.csv` and plots a two-panel bar chart (Test Accuracy and Test Macro-F1) with `±1` sample SD (`ddof=1`) error bars, saving `results/figures/transfer_vs_scratch_comparison.png`. | Visual model comparison | Gives a clear visual comparison of frozen vs full fine-tuning alongside the Part B scratch CNN for the report and viva. |

---

## 3. Concepts I learned

### Concept 1: Transfer Learning and Domain Shift (ImageNet vs EuroSAT)
- **What it is:** Transfer learning reuses a model trained on a large dataset (ImageNet natural photos) as the starting point for a different task (EuroSAT satellite patches), where domain shift is the difference between everyday ground-level photos (`224x224`) and top-down satellite images (`64x64`).
- **Where in our code:** `src/models.py` in `build_transfer_model()` (`ResNet18_Weights.DEFAULT` and `MobileNet_V2_Weights.DEFAULT`).
- **Likely question and my answer:** *Why do ImageNet weights help on satellite images if ImageNet has no satellite classes?* Early CNN layers detect basic edges, corners, colour contrasts and simple textures that appear in both natural photos and satellite imagery. Deeper layers are more specific to ImageNet objects, which is why updating all layers (full fine-tuning) worked much better on EuroSAT than keeping the backbone frozen.

### Concept 2: Frozen Feature Extraction vs Full Fine-Tuning (and `requires_grad=False`)
- **What it is:** In frozen feature extraction, we lock the pretrained backbone weights (`param.requires_grad = False`) and only train the new 10-class linear layer; in full fine-tuning (`freeze=False`), all weights in the network stay trainable (`requires_grad = True`) and adapt to EuroSAT.
- **Where in our code:** `src/models.py` lines 66-69 (`for param in model.parameters(): param.requires_grad = False`) and `src/train.py` line 165 (`[p for p in model.parameters() if p.requires_grad]`).
- **Likely question and my answer:** *What actually gets updated when `freeze=True`?* Only the weights and biases of the new 10-class linear head (`5,130` parameters in `resnet18_frozen` and `12,810` in `mobilenet_v2_frozen`) receive gradient updates from Adam.

### Concept 3: BatchNorm Running Statistics When the Backbone Is Frozen
- **What it is:** BatchNorm layers have two learnable parameters (scale and shift, which are stopped by `requires_grad=False`) and two running buffers (`running_mean` and `running_var`), which are not `nn.Parameter` tensors and still update whenever `model.train()` is active.
- **Where in our code:** `src/train.py` line 175 calls `model.train()` on the whole model every epoch.
- **Likely question and my answer:** *If `freeze=True`, is the backbone 100% static during training?* Not completely. Its learnable weights are frozen by `requires_grad=False`, but because `src/train.py` calls `model.train()` on the whole network, the `BatchNorm2d` layers in the backbone still update their `running_mean` and `running_var` buffers using EuroSAT batch statistics.

### Concept 4: Differential Learning Rates (`0.001` vs `0.0001`) and Confounding
- **What it is:** We use a standard learning rate (`0.001`) when training only the new linear head from scratch, and a 10x smaller learning rate (`0.0001`) during full fine-tuning so large gradient steps do not wreck the pretrained backbone weights.
- **Where in our code:** `src/train.py` line 61 (`return models.build_transfer_model(backbone, freeze), (1e-3 if freeze else 1e-4)`).
- **Likely question and my answer:** *Can we say the +11.14 percentage-point jump from `resnet18_frozen` to `resnet18_full` is caused only by unfreezing the backbone?* No. Because both trainability (`freeze`) and learning rate (`0.001` vs `0.0001`) changed together between the two configurations, the improvement is the combined effect of the full fine-tuning configuration (`freeze=False` at `lr=0.0001`) compared with the frozen configuration (`freeze=True` at `lr=0.001`).

### Concept 5: ResNet-18 Residual Connections vs MobileNetV2 Inverted Residuals & Depthwise Separable Convolutions
- **What it is:** ResNet-18 uses skip (shortcut) connections that add a block's input directly to its output so gradients flow easily through deep layers, while MobileNetV2 uses depthwise separable convolutions (splitting spatial filtering and channel mixing) and inverted residual blocks to keep parameter counts small (`2.24M` vs `11.18M`).
- **Where in our code:** Loaded in `src/models.py` lines 59-62 and compared in `notebooks/03_transfer_learning.ipynb`.
- **Likely question and my answer:** *Did `mobilenet_v2_full` (`98.27%`) beat `resnet18_full` (`97.98%`) in a statistically proven way?* No. The difference is only `0.29` percentage points (about 12 test images out of 4,050) across 3 seeds on one fixed split. Both full fine-tuning models achieved roughly `98%` accuracy, while MobileNetV2 used about 5x fewer parameters.

### Concept 6: Sample Standard Deviation (`ddof=1`) Across Three Seeds
- **What it is:** Sample standard deviation (`ddof=1`) divides the sum of squared deviations by $n - 1$ instead of $n$, correcting for the bias of estimating spread from a small sample ($n = 3$ seeds).
- **Where in our code:** `notebooks/03_transfer_learning.ipynb` in `cell-10-lap2-summary-table` (`acc_pct.std(ddof=1)`) and `cell-12-comparison-plot`.
- **Likely question and my answer:** *Why did we switch from `ddof=0` to `ddof=1` in the summary table?* Three random seeds are a small sample of possible training runs, and Part B reported the `scratch_cnn` baseline (`92.72% ± 0.45%`) using `ddof=1`. Using `ddof=1` for all five models keeps our statistical comparison consistent.

---

## 4. What I understood from the previous part

Before starting my Part C transfer-learning experiments, I inspected Srivalli's data loader (`src/dataset.py`), Tejash's Part B files (`src/models.py`, `notebooks/02_baseline_cnn.ipynb`, `report/part_b_scratch_cnn_draft.md`), and Nitin's shared training script (`src/train.py`):

- **Tejash's `build_scratch_cnn` (`src/models.py`):** Builds a compact custom CNN (`93,962` parameters) for `64x64` RGB patches using 3 convolutional blocks (`32 -> 64 -> 128` filters). Each block runs `Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)`, followed by `AdaptiveAvgPool2d(1)`, `Flatten()`, optional `Dropout`, and `Linear(128, 10)`.
- **Purpose of BatchNorm and Part B ablations:** BatchNorm normalises feature activations inside each mini-batch so training is more stable. Tejash ran 9 experiments (`3 configurations × seeds 0, 1, 2`): the 3-block baseline achieved **92.72% ± 0.45%** test accuracy (**92.48% ± 0.51%** macro-F1), removing BatchNorm (`scratch_cnn_no_batchnorm`) dropped accuracy to **88.95% ± 0.58%**, and reducing depth to 2 blocks (`scratch_cnn_depth2`) dropped accuracy to **85.31% ± 1.43%**.
- **Nitin's shared `src/train.py` and how Part B connects to Part C:** `src/train.py` caches all 27,000 EuroSAT images in memory, uses the exact same `data/splits.csv` (`70/15/15` split), ImageNet normalisation, training-only rotations/flips, `Adam(..., foreach=False)`, batch size `64`, and 20 epochs, picks the best checkpoint using validation accuracy only, and scores the test set once to update `results/metrics.csv`. Part B gave me the **92.72% ± 0.45%** scratch CNN baseline on the same test split so I could fairly test whether frozen or fine-tuned ResNet-18 and MobileNetV2 improve over training a small CNN from scratch.

---

## 5. Sources I used

- **EuroSAT benchmark paper:** Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth (2019), *"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,"* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), Vol. 12, No. 7, pp. 2217-2226. DOI: https://doi.org/10.1109/JSTARS.2019.2918242 (arXiv: https://arxiv.org/abs/1709.00029)
- **Transfer learning in remote sensing:** Keiller Nogueira, Otavio A. B. Penatti, and Jefersson A. dos Santos (2017), *"Towards better exploiting convolutional neural networks for remote sensing scene classification,"* Pattern Recognition, Vol. 61, pp. 539-556. DOI: https://doi.org/10.1016/j.patcog.2016.07.001 (arXiv: https://arxiv.org/abs/1602.01517)
- **Remote sensing scene classification survey:** Gong Cheng, Junwei Han, and Xiaoqiang Lu (2017), *"Remote Sensing Image Scene Classification: Benchmark and State of the Art,"* Proceedings of the IEEE, Vol. 105, No. 10, pp. 1865-1883. DOI: https://doi.org/10.1109/JPROC.2017.2675998 (arXiv: https://arxiv.org/abs/1703.00121)
- **ResNet architecture paper:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016), *"Deep Residual Learning for Image Recognition,"* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778. DOI: https://doi.org/10.1109/CVPR.2016.90 (arXiv: https://arxiv.org/abs/1512.03385)
- **MobileNetV2 architecture paper:** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen (2018), *"MobileNetV2: Inverted Residuals and Linear Bottlenecks,"* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520. DOI: https://doi.org/10.1109/CVPR.2018.00474 (arXiv: https://arxiv.org/abs/1801.04381)
- **Official PyTorch & Torchvision documentation:** `torchvision.models.resnet18`, `torchvision.models.mobilenet_v2`, `torch.nn.Linear`, `torch.nn.BatchNorm2d`, and PyTorch autograd `requires_grad` mechanics (https://pytorch.org/vision/stable/models.html and https://pytorch.org/docs/stable/).
- **AI tools used:**
  - **ChatGPT:** Used to break down `docs/plan.md` into smaller subtasks, understand transfer-learning and statistical concepts, prepare step-by-step instructions, and review experimental results and report conclusions.
  - **Antigravity (Gemini):** Used for repository inspection, implementing `build_transfer_model` in `src/models.py`, building and updating `notebooks/03_transfer_learning.ipynb`, recovering interrupted Lap 2 logs into `transfer_all_seeds_histories.json`, generating the `ddof=1` statistical table and comparison plot (`transfer_vs_scratch_comparison.png`), and drafting `report/part_c_transfer_learning_draft.md` and my log/viva/AI disclosure entries.

---

## 6. My 6 viva questions

1. **Question:** What is transfer learning and why did we use it for EuroSAT satellite images?
   **Answer:** Transfer learning starts from a deep CNN (ResNet-18 or MobileNetV2) already pretrained on 1.28 million ImageNet photos instead of random weights. We used it because early CNN layers already know how to detect edges, colour contrasts and textures, letting us test whether adapting a deep pretrained network beats training our small `93,962`-parameter scratch CNN from random initialisation.
2. **Question:** What is the difference between our frozen-backbone and full fine-tuning configurations?
   **Answer:** In the frozen configurations (`resnet18_frozen`, `mobilenet_v2_frozen`), we set `requires_grad = False` on the pretrained backbone so only the new 10-class linear head trains (`5,130` and `12,810` parameters) at `lr = 0.001`. In the full fine-tuning configurations (`resnet18_full`, `mobilenet_v2_full`), all layers stay trainable (`11.18M` and `2.24M` parameters) and update together at a 10x smaller learning rate (`lr = 0.0001`) so we do not destroy the pretrained features.
3. **Question:** How did you modify ResNet-18 and MobileNetV2 in `src/models.py` to output 10 EuroSAT classes?
   **Answer:** Both pretrained models originally output 1,000 ImageNet classes. In `build_transfer_model()`, after optionally freezing the backbone, I replaced ResNet-18's final layer `model.fc` with `nn.Linear(512, 10)` and replaced MobileNetV2's final layer `model.classifier[1]` with `nn.Linear(1280, 10)`. Because a new `nn.Linear` layer defaults to `requires_grad = True`, the new 10-class head is always trainable.
4. **Question:** Why did the frozen models perform worse than both the fully fine-tuned models and the scratch CNN?
   **Answer:** `resnet18_frozen` (`86.85% ± 0.32%`) and `mobilenet_v2_frozen` (`87.70% ± 0.29%`) trailed the scratch CNN (`92.72% ± 0.45%`), while `resnet18_full` (`97.98% ± 0.14%`) and `mobilenet_v2_full` (`98.27% ± 0.12%`) improved accuracy by `+11.14 pp` and `+10.58 pp` over their frozen versions. There is a domain shift between ground-level `224x224` ImageNet object photos and top-down `64x64` satellite patches; keeping the deeper convolutional filters fixed limits how well the features separate overhead land-use classes, whereas full fine-tuning adapts all convolutional filters to EuroSAT (noting that frozen and full also use different learning rates: `0.001` vs `0.0001`).
5. **Question:** What does BatchNorm do, and why do its running statistics still update when `freeze=True`?
   **Answer:** BatchNorm normalises each channel's activations using the batch mean and variance and applies learnable scale and shift parameters. Setting `param.requires_grad = False` freezes the learnable scale and shift weights, but `running_mean` and `running_var` are stored as buffers, not parameters. Because `src/train.py` calls `model.train()` on the whole network every epoch, those BatchNorm running buffers continue updating on EuroSAT batches even in frozen mode.
6. **Question:** What did the three-seed results show, and why did you use sample standard deviation (`ddof=1`)?
   **Answer:** Running seeds 0, 1 and 2 showed very small spread (`0.12%` to `0.32%` SD for the transfer models), meaning training is stable across different random head initialisations, batch shuffle orders and augmentation draws on our fixed split. However, the `+0.29 pp` gap between `mobilenet_v2_full` (`98.27%`) and `resnet18_full` (`97.98%`) is too small across 3 seeds to claim one architecture is statistically superior. We used sample standard deviation (`ddof=1`, dividing by $n - 1$) because 3 seeds are a small sample and it matches Part B's baseline formula.

---

## 7. AI Session Report (Section 6A Record)

### Student's Three Section 6A Answers (Recorded Faithfully)

#### 1. What I personally ran or checked
I personally performed the following setup, execution and verification activities during my Part C work:
- Checked my Python environment using `python --version`, `where.exe python` and `pip show torch torchvision`. I found multiple Python installations and selected the correct Python 3.12.10 environment for the native Jupyter kernel.
- Verified PyTorch and CUDA using `torch.__version__`, `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`. I confirmed PyTorch 2.14.1+cu130, CUDA availability and my NVIDIA GeForce RTX 4050 Laptop GPU.
- Ran `python check_setup.py --name Rachana --download` to check the project setup, dataset and pretrained weights.
- Inspected `src/models.py`, `src/train.py`, `src/dataset.py`, notebook 02 and teammate logs to understand the existing scratch CNN and shared training pipeline.
- Executed the four seed-0 transfer-learning configurations in the native Jupyter kernel and checked their validation and test results.
- After a CUDA driver error interrupted the second experiment batch, I restarted the Jupyter kernel, reran the setup cells and executed the remaining `mobilenet_v2_full` seed-2 experiment using the notebook's skip logic.
- Checked `results/metrics.csv` and confirmed that it contained 9 Part B rows and 12 Part C rows. I also checked the 12 transfer-learning checkpoints.
- Reviewed the three-seed summary, sample standard deviation calculations, learning curves and comparison chart.

I used ChatGPT to break the work into smaller tasks and understand the concepts. I used Gemini in Antigravity to assist with implementation, notebook execution, debugging and documentation. I reviewed the outputs and requested corrections where needed.

#### 2. Which lines or concepts I initially found difficult or still need to understand better
I initially needed clarification on:
- How ResNet-18 and MobileNetV2 are loaded with pretrained weights and how their final classification layers are replaced for our 10 EuroSAT classes.
- The difference between freezing a pretrained backbone and performing full fine-tuning.
- How `requires_grad=False` works and which parameters are updated during training.
- Why full fine-tuning uses a smaller learning rate than frozen feature extraction.
- How BatchNorm running statistics can continue updating even when the backbone's learnable parameters are frozen.
- The difference between population standard deviation (`ddof=0`) and sample standard deviation (`ddof=1`).
- Why CPU and GPU augmentation implementations may not produce identical random transformations, even when the same seeds and probabilities are used.

I understand the main transfer-learning workflow better now. I still want to strengthen my understanding of BatchNorm running statistics, training reproducibility and the differences between frozen feature extraction and full fine-tuning.

#### 3. What I changed or questioned in the AI's suggestions
I did not blindly accept the generated code or conclusions. I divided the work into smaller tasks, reviewed the outputs and requested corrections whenever something needed clarification.
- I asked for separate implementation and verification steps for ResNet-18 and MobileNetV2.
- I checked the model outputs and trainable parameter counts before proceeding with the experiments.
- I requested that completed experiments be skipped instead of unnecessarily retrained, so existing results would not be overwritten.
- After the CUDA interruption, I restarted the kernel and used the skip logic to run only the missing experiment.
- I requested the use of sample standard deviation (`ddof=1`) to match the Part B baseline.
- I questioned the frozen-versus-full comparison because both trainability and learning rate change between those configurations. I requested that the report avoid attributing the entire improvement only to unfreezing.
- I requested a clearer explanation of BatchNorm running statistics when `freeze=True`.
- I asked to keep the Related Work section focused on Part C topics, leaving Grad-CAM and t-SNE to the teammate responsible for explainability.
- I instructed the AI not to overwrite Part B results, delete files or retrain completed experiments during the final checks.

My contribution was directing the workflow, performing the environment checks and experiment execution assigned to me, reviewing the results and requesting corrections. AI tools assisted with code generation, debugging, analysis and documentation, and I reviewed their outputs throughout the process.

```
=== AI SESSION REPORT ===
Date: 2026-11-04
Student: Rachana (Part C, baton 4)
AI tool and model: ChatGPT (for task breakdown, concept explanation, and review prompts) and Antigravity (Gemini, for repository inspection, code/notebook implementation, log recovery, statistical plotting, and report/log drafting)
Topic: Part C Transfer Learning (ResNet-18 & MobileNetV2 frozen vs full fine-tuning across 3 seeds, comparison with scratch CNN, report draft, and notebook rerun safety)
What the student asked, in order:
1. Inspect repository setup, Part B scratch CNN baseline, and shared training pipeline (src/train.py).
2. Implement build_transfer_model(name, freeze) in src/models.py for resnet18 and mobilenet_v2 and verify parameter counts.
3. Build notebooks/03_transfer_learning.ipynb and run Lap 1 (seed 0) and Lap 2 (seeds 0, 1, 2) transfer-learning experiments.
4. Recover the 7 completed Lap 2 epoch histories from task-594.log after a CUDA display driver reset and complete the 8th Lap 2 run (mobilenet_v2_full seed 2) in Jupyter.
5. Update three-seed statistical calculations to sample standard deviation (ddof=1) to match the Part B baseline.
6. Generate the five-model comparison chart (results/figures/transfer_vs_scratch_comparison.png) and draft report/part_c_transfer_learning_draft.md with corrected rounding, non-causal learning-rate/trainability wording, and BatchNorm running-buffer explanation.
7. Add 3-layer artifact validation and skip guards to cell-04-run-seed0 and cell-08-run-lap2 in notebooks/03_transfer_learning.ipynb and complete Part C documentation in docs/log/rachana.md, docs/ai_usage.md, and docs/viva.md.
What the AI wrote:
- src/models.py: build_transfer_model(name, freeze)
- notebooks/03_transfer_learning.ipynb: configuration table, seed-0 and multi-seed training/validation loops with 3-layer artifact skip guards, seed-0 learning curve plot, 12-run history verification table, ddof=1 statistical summary table, and 5-model comparison chart
- results/figures/transfer_all_seeds_histories.json, results/figures/transfer_learning_curves_seed0.png, results/figures/transfer_vs_scratch_comparison.png
- report/part_c_transfer_learning_draft.md
- Documentation entries in docs/log/rachana.md, docs/ai_usage.md, and docs/viva.md
What the student wrote or changed:
- See Answer 3 above (requested separate verification steps for ResNet-18 and MobileNetV2, checked parameter counts, enforced skip logic to protect completed runs, restarted kernel after CUDA interruption to run only the missing mobilenet_v2_full seed-2 run, enforced sample SD ddof=1, required non-causal wording for frozen vs full fine-tuning due to learning-rate confounding, requested BatchNorm running-statistics explanation, kept Related Work scoped to Part C topics, and prohibited overwriting Part B results or completed experiments).
NOT RUN (the AI did not actually run these):
- Full fresh-kernel Restart & Run All execution of notebooks/03_transfer_learning.ipynb after adding the final 3-layer skip guards in cell-04 and cell-08 (the saved cell guard code was verified directly in Python against the on-disk artifacts without launching nbclient/nbconvert).
- Optional extra ResNet-18 run at 128x128 resolution (marked optional if time in docs/plan.md).
```
