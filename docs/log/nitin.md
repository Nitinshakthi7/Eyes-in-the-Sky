# Work log: Nitin (Part D - Repo, train(), data-efficiency study, report assembly)

**Baton:** 1 and 5
**Files I own:** `src/utils.py`, `src/train.py`, `notebooks/05_data_efficiency.ipynb`, `check_setup.py`, `results/metrics.csv`, README
**My task card:** [docs/plan.md](../plan.md), section 4, "Baton 1 and 5: Part D, Nitin"

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
| 2026-11-02 | Set up the repo: skeleton, `check_setup.py`, team plan, member logs. Merged PR 1. | With Claude Code. Ran `check_setup.py` on my laptop (all checks OK, RTX 3050). Tested the PyTorch install in a clean venv. Made 2 commits, pushed, opened and merged the pull request. | [PR 1](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/1) | Teammates not added as collaborators yet |
| 2026-11-02 | Baton 1: wrote `src/utils.py` (seed, device, paths) | With Claude Code. Wrote 3 small helpers, then a self-check at the bottom of the file. Ran `python -m src.utils`: passed. Wrote the notes below. | [PR 2](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/2) | I still have to read every line below and mark what I do not understand yet |
| 2026-11-03 | Baton 5: wrote `src/train.py` (the shared `train()`), earlier than planned because Tejash had already finished | With Claude Code. Read Tejash's pushed branch first and checked his settings against the team settings. Wrote `train()` from his helper, plus a `fraction` option, all images kept in memory, and a results file that replaces a repeated run's row. Tested it on the real data: every check passed, and it reproduced his baseline (mean of 3 seeds 92.79% against his 92.72%) at 4.6 s per epoch. Wrote the notes below. | [PR 7](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/7) (branch `d-train-nitin`) | Nitin's answers: understands it all ("I understand it all"), no changes to the AI's work. He first said "None of these yet", then ran `python -m src.train` himself and it passed; asked to read the code and the notes, he then said: "I have read and understood" (2026-11-03) |
| 2026-11-04 | Wrote and ran the data-efficiency study (`notebooks/05_data_efficiency.ipynb`): scratch CNN against MobileNetV2 (fully fine-tuned) at 5%, 10%, 25%, 50% and 100% of the training images, seeds 0 to 2 | With Claude Code. Built the notebook around `train()`. It reuses the 100% rows that already exist and skips finished runs. Ran the 24 missing runs on my RTX 3050 in 20 minutes with the team settings. Checked that `results/metrics.csv` only gained rows (21 old rows unchanged, no duplicates) and saved the plot `results/figures/data_efficiency.png`. Wrote the notes below and the open-items list (section 7) | [PR 9](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/9) (branch `d-data-efficiency-nitin`) | Nitin's own check of this notebook: NOT CHECKED YET. Seeds 3 and 4 (Srivalli, Rohan) are still missing |
| 2026-11-05 | Checked Tejash's checkpoints, reviewed and fixed Rohan's PR 12, filled the Part E gaps (notebook 04 section 9), updated the README and drafted the Part E report section | With Claude Code. Opened Tejash's 2.3 GB zip safely (listed it first, took out only the 9 `.pth` files, loaded each with `weights_only=True`, re-scored the test set: all 9 match his rows in `metrics.csv` exactly). Reviewed Rohan's PR 12 on a clean copy: his numbers and seed-4 rows are right (his 5% row reproduced within 0.25 points), but his Grad-CAM conclusions had no control, 3 scripts only ran as `python src/x.py`, and the notebook only ran from `notebooks/`. Fixed these in a commit on his branch (added `src/gradcam_sanity_check.py`, corrected the wording with marked notes) and merged PR 12 (merge commit 167b94d). Then merged my own notebook 04 work into his: his notebook stays, my extras are its section 9. Wrote `report/part_e_error_analysis_and_explainability_draft.md` from the real numbers and updated the README | [PR 12](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/12) merged. My own changes are on the local branch `e-part-e-completion-nitin` (not committed or pushed yet) | Nitin's own check of all this: NOT CHECKED YET. Rohan has not seen the corrections or my section 9. My first hide-a-patch check was unfair (scattered pixels); the AI caught it and redid it |

## 2. Code notes (line by line)

*These notes were drafted by Claude Code from the finished file. Nitin: read them next to the code, fix anything wrong, and write "I do not understand this yet" beside anything that is still unclear.*

### File: `src/utils.py` / Function: all of it (imports, paths, `set_seed`, `get_device`, self-check)

Why this file exists: three things are needed by every part (Srivalli's data code, Tejash's and Rachana's training, my `train()`, Rohan's evaluation): the same random seed method, the same way to pick GPU or CPU, and the same folder names. If everyone writes their own, results stop being comparable.

**Imports and paths (lines 14 to 29)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 14 | `import random` loads Python's built-in random-number tool | Standard library | `set_seed` has to seed this one too, because some code (for example when picking random images) uses it |
| 15 | `from pathlib import Path` loads a tool that treats a file path as an object | `pathlib` | `/` joins path parts safely on Windows, so we avoid strings full of backslashes. We could have used `os.path.join`, which is older and clumsier |
| 17 | `import numpy as np` loads numpy, the number-array library | numpy | scikit-learn (the split) and pandas use numpy's random numbers, so we seed numpy too |
| 18 | `import torch` loads PyTorch | PyTorch | It is our deep-learning library. The seed and the GPU check both come from it |
| 23 | `ROOT = Path(__file__).resolve().parent.parent` finds the repo folder. `__file__` is the address of this file. `.resolve()` makes it a full address. Each `.parent` goes up one folder: `src/utils.py` to `src` to the repo folder | Path relative to the file | It works on any laptop and in any folder. If we typed `C:\Users\Nitinshakthi\...` it would only work on mine |
| 24 to 27 | `DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `FIGURES_DIR` are the four folders we use. `ROOT / "data"` means "the data folder inside the repo" | Constants (names that never change) | One place to fix a folder name, and nobody types a path by hand. Our folder name has spaces, which breaks hand-typed paths easily |
| 28 | `SPLITS_CSV` is the file with the fixed 70/15/15 split | Constant | Srivalli writes this file in baton 2; everyone reads it from here |
| 29 | `METRICS_CSV` is the results table | Constant | `train()` will add one row per run to this file. The brief requires all results in `results/metrics.csv` |

**`set_seed` (lines 32 to 39)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 32 | `def set_seed(seed):` makes a function. `seed` is any whole number | Function, random seed | We will use 0, 1, 2, 3 and 4 (and 42 for the split) |
| 33 | The text in triple quotes explains the function | Docstring | So anyone can read what it is for without reading the code |
| 34 | `random.seed(seed)` fixes Python's random numbers | Random seed | After this, "random" choices come out in the same order every time |
| 35 | `np.random.seed(seed)` fixes numpy's random numbers | Random seed | Same idea for numpy (used by scikit-learn and pandas) |
| 36 | `torch.manual_seed(seed)` fixes PyTorch's random numbers on CPU and GPU | Random seed | PyTorch uses random numbers for the starting weights, dropout, the order of the batches and augmentation. Fixing it makes a run repeatable |
| 37 | `cudnn.deterministic = True` tells the GPU to use the same convolution method every time | cuDNN (NVIDIA's library of fast convolution routines) | Some methods give tiny different answers each run. This setting can be slightly slower, which is fine for us. We could skip it and accept small differences |
| 38 | `cudnn.benchmark = False` stops the GPU from testing several methods and picking the fastest | cuDNN benchmark | The fastest method can change from run to run, which would break repeatability |
| 39 | A comment: two laptops can still give slightly different numbers | Reproducibility limits | Different GPUs and driver versions round decimals a little differently. In the report we say "seeds fixed, small differences between machines are possible" and we report mean and spread over several seeds |

**`get_device` (lines 42 to 44)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 42 to 43 | A function and its docstring | Function | Every part calls `get_device()` instead of deciding by itself |
| 44 | `torch.cuda.is_available()` asks "can PyTorch use an NVIDIA GPU?". If yes the function returns `"cuda"` (the GPU), otherwise `"cpu"` | GPU vs CPU, device | Later we write `model.to(device)` and `images.to(device)`. The same code then runs on a GPU laptop and on a CPU laptop (Srivalli's and Rohan's) without changes |

**Self-check (lines 47 to 61)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 47 | `if __name__ == "__main__":` means "only run the lines below when this file is started directly" | `__main__` guard | When another file does `from src.utils import set_seed`, this block must NOT run |
| 49 to 54 | Makes three small lists of random numbers: seed 0, seed 0 again, seed 1 | Testing a seed | The first two must be identical and the third must differ |
| 55 and 56 | `assert` stops the program with a message if the statement is false | Assertion | A very small test: if someone breaks `set_seed`, running the file shows it right away. A proper test framework would be more than we need |
| 57 and 58 | Checks that the four folders really exist | Assertion | Catches a wrong path early |
| 59 to 61 | Prints the result, which device was found, and the repo folder | Output | Easy for any teammate to see that it worked |

**Result of the self-check on my laptop (2026-11-02):** exit code 0; `set_seed: OK`; `get_device: cuda`; seed 0 twice gave `[0.4962..., 0.7682..., 0.0884...]` both times and seed 1 gave `[0.7576..., 0.2793..., 0.4030...]`.

### File: `src/train.py` / Function: all of it (imports, model names, fast data, scoring, saving, `train`, self-check)

Why this file exists: the brief wants ONE reusable training function, so every experiment (Tejash's scratch CNN, Rachana's transfer models, my data-efficiency study) is trained and recorded the same way. I based it on Tejash's notebook helper, because his settings already matched the team settings. I added the `fraction` option and made it much faster by keeping all the images in memory.

**Imports, settings and model names (lines 20 to 62)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 20 to 22 | `tempfile` makes a throw-away folder (only the self-check uses it). `time` is a stopwatch. `Path` handles file paths | Standard library | The self-check must never touch the real results file |
| 24 to 28 | numpy and pandas (arrays and tables), torch (PyTorch), `Image` (opens a picture file), `f1_score` (the macro-F1 number) | Libraries | `f1_score` comes from scikit-learn, so we do not write our own |
| 30 | `from src import models` loads Tejash's file, which has `build_scratch_cnn` | Import | `train()` builds a model by name instead of copying its code |
| 31 | Takes the ImageNet mean/std and `get_loader` from Srivalli's `dataset.py` | Reuse | The same normalisation numbers for all models, and `get_loader` is used to pick exactly the same images as her code |
| 32 | Takes the folders, the split file, `get_device` and `set_seed` from my `utils.py` | Reuse | One place for paths and seeds |
| 34 and 35 | `BATCH_SIZE = 64` and `MAX_EPOCHS = 20` | Team settings (docs/plan.md, section 4) | One epoch is one pass over all the training images. A batch is the small group of images used for one weight update |
| 38 to 42 | `SCRATCH` links three scratch model names to the switches of `build_scratch_cnn` (BatchNorm on or off, 3 or 2 blocks) | Dictionary | These are the names Tejash used in `metrics.csv`, so his rows and mine match |
| 44 to 49 | `TRANSFER` links four transfer names to (backbone, frozen or not) | Dictionary, frozen vs full | This is Rachana's part. Frozen means only the new last layer learns |
| 52 to 62 | `build_model` returns the model and its learning rate. Scratch: 0.001. Transfer: 0.001 when frozen, 0.0001 when fully fine-tuned. A transfer name before Rachana's function exists gives a clear message, and an unknown name gives a clear error | Learning rate | A lower rate for full fine-tuning so we do not damage the pretrained weights (plan, card C). We could have used one rate for everything, but that is a worse choice for pretrained models |

**Fast data: all images in memory (lines 65 to 106)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 66 | `_cache = {}` is a box that remembers the loaded images | Cache | The first `train()` call fills it, and every later call in the same Python session reuses it |
| 69 to 80 | `_load_all_images` reads the 27,000 pictures once, turns them into one block of numbers of shape (27000, 3, 64, 64), and puts it on the GPU. It also keeps the labels and a "which row is this file" lookup | In-memory data (bytes, 0 to 255) | Reading and decoding a JPEG for every image in every epoch was slow (about 12 s per epoch just to read, on my laptop). Kept as bytes it is only about 330 MB. We could have used a DataLoader with several workers, but that is fragile on Windows |
| 83 to 87 | `_rows` gives the row numbers of one split. For train it asks Srivalli's `get_loader` which images the `fraction` keeps | Stratified subset | We reuse her exact rule, so the 5% subset is the same images as her code would pick. No images are read in this step |
| 90 to 92 | `_prepare` turns bytes into decimals between 0 and 1 (divides by 255) | Scaling pixels | The same first step as `ToTensor` in her code |
| 93 to 103 | With `augment`: each image gets a random turn of 0, 90, 180 or 270 degrees, then a random sideways flip and a random upside-down flip, using masks so the whole batch is done at once | Augmentation (label-safe for satellite images) | A satellite image has no "up", so a turned or flipped field is still a field. Doing it on the GPU is fast. It makes the same kind of changes as `augment()` in `dataset.py` |
| 104 to 106 | Subtracts the ImageNet mean and divides by the ImageNet std for each colour channel | Normalisation | The team uses the same numbers for all models, so the comparison is fair |

**Scoring and saving the results (lines 109 to 136)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 109 to 111 | `_score` puts the model in evaluation mode | `model.eval()` | Evaluation mode switches off training-only behaviour (BatchNorm uses its saved averages instead of the batch's) |
| 114 to 119 | Goes through the rows in groups of 256 with no gradient tracking, adds up the loss, and collects the predicted class (the biggest score) | `torch.no_grad()`, argmax | We do not train here, so no gradients are needed. That saves memory and time. 256 is bigger than 64 because nothing is learned here, so it makes no difference to the answer |
| 120 to 122 | Accuracy is the share of predictions that equal the true labels. Returns average loss, accuracy and predictions | Accuracy | |
| 125 to 136 | `_save_row` adds one row to the results file. If a row for the same model, fraction and seed already exists, it is replaced | Replace instead of duplicate | Tejash's nine rows are already in `metrics.csv`. A rerun must not create duplicates. The file keeps exactly the 5 columns the brief asks for |

**`train` (lines 139 to 227)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 139 | `def train(model_name, fraction, seed, ...)` is the function the brief asks for | Function with arguments | The other arguments have defaults (20 epochs, the real results file, save weights only for full data) |
| 150 and 151 | Stops with an error if `fraction` is not between 0 and 1 | Input check | A wrong value should fail at once, not after an hour |
| 152 and 153 | Picks the device and sets the seed | Seed, device | From `utils.py`. The seed comes first so everything after it is repeatable |
| 155 to 158 | Gets the images and works out which rows are train, validation and test | Fixed split | Validation and test never shrink. Only the train rows follow `fraction` |
| 160 and 161 | Builds the model by name and moves it to the device | `model.to(device)` | The same code works on GPU and CPU |
| 162 | `CrossEntropyLoss` | Loss function | The standard loss for choosing one class out of several. It is the same as Tejash's |
| 165 and 166 | `Adam` with the learning rate from `build_model`. Only the layers that are allowed to learn are given to it | Optimiser, `requires_grad` | Frozen layers must not be updated. `foreach=False` is the same setting Tejash used |
| 167 | `shuffler` is a random generator started from the seed | Seeded shuffle | It decides the order of the images, so a run can be repeated |
| 169 to 171 | Prepares to remember the history, the best validation accuracy, its epoch and its weights, and starts a stopwatch | Bookkeeping | |
| 173 to 175 | Starts each epoch and switches to training mode | `model.train()` | BatchNorm and dropout behave differently when training |
| 176 to 178 | Counters for the loss and the correct answers, kept on the GPU | Counters | Reading them from the GPU after every batch would slow things down, so they are read once per epoch |
| 180 | Shuffles the training rows | Shuffle | Different order each epoch helps learning |
| 181 to 184 | Takes the rows in groups of 64. A group of only one image is skipped | Batches, BatchNorm | BatchNorm cannot work with a single image. With our sizes this never happens, but the line prevents a crash |
| 185 and 186 | `x` is the augmented, normalised images. `y` is their true labels | Augmentation only in training | |
| 187 to 191 | The five steps of learning: clear the old gradients, run the model, measure the loss, work out the gradients (`backward`), update the weights (`step`) | Training step | This is the core loop of all neural-network training |
| 192 to 194 | Adds up the loss and the number of correct answers | Metrics | |
| 196 to 200 | After the epoch, scores the validation set and stores the four numbers | Validation | This is how we see learning and overfitting |
| 201 to 203 | Prints one line per epoch, with the seconds | Progress | Handy for timing a laptop |
| 205 to 207 | If this epoch's validation accuracy is the best so far, remember the epoch and a copy of the weights | Best-epoch selection on validation | The best epoch is chosen on VALIDATION, never on test |
| 209 | Puts the best epoch's weights back into the model | `load_state_dict` | We report the best version, not the last one |
| 211 to 213 | Scores the test set once and computes accuracy and macro-F1 | Test set, macro-F1 | Used once and nothing is decided from it. Macro-F1 gives each class equal weight, so a weak class is not hidden |
| 214 and 215 | Makes the result row and saves it | Results file | One row per run, as the brief asks |
| 217 to 221 | Saves the best weights in `models/`, by default only when `fraction` is 1.0 | Checkpoint | Rohan needs the full-data models. The folder is ignored by git, so no weights are committed |
| 223 to 227 | Prints the final line and returns the result and the history | Return value | Notebooks can use the history to plot curves |

**Self-check (lines 230 to 239)**

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 230 to 239 | Trains for 2 epochs on 5% of the data into a temporary file, checks one row was written with a sensible accuracy, runs it again, and checks that there is still only one row | `__main__` guard, assertion | Run it with `python -m src.train`. It never touches the real `metrics.csv` |

**How it was tested (2026-11-03):** the self-check passed. A longer test script (kept outside the repo) also passed. It showed that the in-memory pixels equal Srivalli's loader's pixels, that every augmented image is one of the 8 turns or flips of the original, that the 5% subset is exactly the one her `get_loader` picks, and that the train, validation and test rows do not overlap. Full 20-epoch runs reproduced Tejash's results: baseline mean over seeds 0 to 2 was 92.79% (his 92.72%), and the order baseline, no BatchNorm, two blocks came out the same. Speed: about 4.6 s per epoch on my RTX 3050, against 45 to 90 s on his laptop.

### File: `notebooks/05_data_efficiency.ipynb` / all of it (settings, sweep, table, plot, gap)

Why this notebook exists: the brief asks for a data-efficiency study with plots and several seeds at small data sizes. It compares our scratch CNN with the best transfer model on validation (MobileNetV2, fully fine-tuned). Everything is trained through the shared `train()`, so the numbers are comparable with Parts B and C.

| Cell | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| 1 (setup) | Finds the repo folder, then imports `train`, the results path and the device | Imports | The same start-up trick as the other notebooks, so it works from the repo folder or from `notebooks/` |
| 2 (settings) | `MODELS_TO_COMPARE`, `FRACTIONS` (5%, 10%, 25%, 50%, 100%), `SEEDS` (0, 1, 2), `TRAIN_SIZE` (18,900), `SIZES` (945 to 18,900) | Constants | One place to change the study. The sizes are turned into numbers of images for the plot's x-axis |
| 4 (sweep) | `already_done` checks whether `metrics.csv` already has a row for a (model, fraction, seed). The loop goes through every seed, model and size, skips finished ones and otherwise calls `train(...)`. The per-epoch printing is hidden and one summary line per run is shown | Resumable loop, `redirect_stdout` | A stopped notebook can be started again. The 100% rows from Parts B and C are reused instead of being trained again (the plan says the 100% points reuse the baseline runs). Looping seed by seed keeps the results balanced if it stops early |
| 6 (table) | Reads `metrics.csv`, turns each fraction into a number of images and groups the rows by model and size. For each group it computes the number of seeds, the mean and the sample standard deviation (`ddof=1`) of accuracy and macro-F1 | Mean and sample SD | The brief asks for several seeds, not one lucky run. `ddof=1` is the correct SD for a small sample of seeds. With a single seed the SD cannot be computed, so it shows as 0 |
| 8 (plot) | Draws test accuracy and macro-F1 against the number of labelled images, one line per model, with error bars. The x-axis is logarithmic. Saves the figure to `results/figures/data_efficiency.png` | Log axis, error bars | The training sizes grow by about 2 to 2.5 times each step, so a log axis spaces them evenly. The error bars show how much the seeds disagree |
| 10 (gap) | Subtracts the scratch CNN's mean accuracy from MobileNetV2's at each size | Gap | Shows directly how much transfer learning helps at each amount of data |
| 11 (text) | The interpretation: transfer learning helps most when labels are scarce (93.69% with 945 images against 92.72% for the scratch CNN with all 18,900), the gap shrinks from 16.6 to 5.6 points, and the limits of the study | Interpretation | Written from the real numbers, including what the study cannot say (same epoch budget for all sizes, only 3 seeds, random split) |

**Result (2026-11-04, my RTX 3050):** 24 runs in 20 minutes. Test accuracy, mean ± sample SD over seeds 0 to 2: scratch CNN 77.07 ± 0.74 (5%), 80.98 ± 1.20 (10%), 84.96 ± 1.86 (25%), 89.45 ± 1.20 (50%), 92.72 ± 0.45 (100%). MobileNetV2 full: 93.69 ± 0.20, 96.15 ± 0.33, 97.26 ± 0.09, 97.76 ± 0.01, 98.27 ± 0.12.

### File: `notebooks/04_evaluation_and_explainability.ipynb` / section 9 (Nitin's extras) and `src/gradcam_sanity_check.py`

Why: Rohan's notebook (sections 1 to 8, merged in PR 11 and PR 12) only displays the files and pictures he saved. The plan also wants the most confused pairs counted and shown with examples, and a check of Grad-CAM. Section 9 and the new script add those without changing his sections. Nothing here trains a model, and Rohan's saved files are not overwritten.

| Cell / file | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| Section 9, setup cell | Loads the checkpoint with `weights_only=True`, scores all 4,050 test images on the CPU, keeps the confidence of every prediction and compares with Rohan's saved matrix (0 cells differ) | `weights_only`, softmax | A check that his files are right, and it gives the confidence used in 9.3 and 9.4 |
| 9.1 | Per-class precision, recall and F1, lowest recall first | Precision, recall | Shows which classes are hard (PermanentCrop 0.960 recall) |
| 9.2 | Counts every (actual -> predicted) pair, biggest first, and the share of mistakes among crop and vegetation classes (36 of 69) | Counting off-diagonal cells | The brief expects some pairs (AnnualCrop/PermanentCrop, Highway/River); we count what our data really shows |
| 9.3 | The first 4 real mistakes of each of the 6 biggest pairs, with the model's confidence. Saves `results/figures/most_confused_pairs.png` | Fixed rule, no cherry-picking | His section 3.1 shows 3 examples without the confidence |
| 9.4 | Confidence when right (0.990) and when wrong (0.763), and how many mistakes fall under 60%, 80%, 90% | Confidence | Shows whether a "not sure" rule could catch mistakes (39 of 69 below 80%, but 60 right answers too) |
| 9.5 | The draft interpretation, marked as written with Claude and not confirmed by Rohan | Interpretation | States the limits: one seed, 69 mistakes, heat maps only 2 x 2 cells |
| `src/gradcam_sanity_check.py` (shown in his section 5.1) | For 200 test images (first 20 of each class) hides 25% of the pixels as one solid patch: the hottest, the coldest or a random 32 x 32 square, and measures the fall in confidence (0.064, 0.061, 0.089). Also counts which quarter of the heat map is hottest (bottom-right 160 of 200, top-left 0) | Hide-a-patch (deletion) check | Grad-CAM is not proof. Rohan's own experiment had no control. A first version used scattered random pixels, which is unfair (scattered noise damages an image far more than one solid patch), so it was redone with patches of equal shape |

**How it was tested (2026-11-05):** the notebook runs top to bottom without errors, from `notebooks/` and from the repo folder (about 1 minute on my CPU). The 3,981 correct matches Rohan's matrix cell for cell. The script reproduced the numbers above on a clean copy. Rohan's saved CSV files were not changed.

## 3. Concepts I learned

### Concept: random seed (reproducibility)

- What it is: a starting number that makes a "random" process come out the same way every time.
- Where in our code: `set_seed` in `src/utils.py`, lines 32 to 39.
- Likely question and my answer: *Why do we set a seed?* Neural networks start from random weights and shuffle the data randomly, so two runs differ. Fixing the seed lets a teammate repeat a run and lets us compare two settings fairly. One seed is still one lucky or unlucky run, so we also run several seeds and report the mean and spread.

### Concept: device (GPU or CPU)

- What it is: the chip that does the calculations. A GPU does many calculations at once, which is much faster for neural networks.
- Where in our code: `get_device` in `src/utils.py`, line 44.
- Likely question and my answer: *What happens on a laptop without a GPU?* `torch.cuda.is_available()` is False, so the function returns the CPU and the same code still runs, only slower.

### Concept: paths relative to the file

- What it is: finding folders from where this file lives, not from a typed address.
- Where in our code: `ROOT` in `src/utils.py`, line 23.
- Likely question and my answer: *Why not write the full path?* It would only work on one laptop. With `__file__` it works wherever the repo is cloned.

### Concept: epoch, batch and the training step

- What it is: an epoch is one pass over all training images. The images are used in small groups called batches (64). After each batch the model's weights are updated a little.
- Where in our code: `train`, lines 173 to 194.
- Likely question and my answer: *What happens in one training step?* Clear the old gradients, run the batch through the model, measure the loss, work out how each weight should change (`backward`), and let the optimiser update the weights (`step`).

### Concept: validation set vs test set (best-epoch selection)

- What it is: the validation set is our practice exam for decisions, such as which epoch to keep. The test set is the final exam, used once.
- Where in our code: lines 205 to 207 (choose on validation) and 211 to 213 (score test once).
- Likely question and my answer: *Why not pick the best epoch on the test set?* Then the test score would be tuned to the test images and look better than it really is. Keeping it untouched makes it an honest estimate.

### Concept: Adam and the learning rate

- What it is: Adam is the optimiser, the rule that updates the weights. The learning rate is the size of each update step.
- Where in our code: lines 165 and 166, and the learning rates in `build_model`, lines 52 to 62.
- Likely question and my answer: *Why 0.0001 for full fine-tuning?* The pretrained weights are already good, so we change them gently. A big step could damage what they learned from ImageNet.

### Concept: keeping the data in memory (why `train()` is fast)

- What it is: instead of opening and decoding 18,900 image files in every epoch, we read all 27,000 once and keep them as numbers.
- Where in our code: `_load_all_images`, lines 69 to 80.
- Likely question and my answer: *How do you know the fast version gives the same results?* We checked that the pixels equal the original loader's pixels, that every augmented image is one of the 8 allowed turns or flips, and that full runs reproduce Tejash's earlier results (baseline mean 92.79% against his 92.72%).

### Concept: macro-F1

- What it is: the F1 score is worked out for each class separately and then averaged, so every class counts equally.
- Where in our code: line 213.
- Likely question and my answer: *Why macro-F1 as well as accuracy?* The classes have different sizes. Accuracy can look good while a small class does badly. Macro-F1 does not hide that.

### Concept: data efficiency (learning with fewer labels)

- What it is: how the score changes as we give the model fewer labelled training images. A model that is "data efficient" stays good with little data.
- Where in our code: notebook 05, the sweep (cell 4) and the plot (cell 8).
- Likely question and my answer: *Why does MobileNetV2 do so well with only 945 images?* It was already trained on a million ImageNet images, so it already knows edges, textures and shapes. Fine-tuning only has to adapt that knowledge to satellite images. The scratch CNN has to learn everything from the 945 images.

### Concept: mean ± sample standard deviation over seeds

- What it is: we repeat each setting with different seeds and report the average and how much the runs disagree. The sample standard deviation (`ddof=1`) divides by one less than the number of runs, which is the right choice for a small sample.
- Where in our code: notebook 05, cell 6 (the table); also in `train()`'s reuse for all parts.
- Likely question and my answer: *Why not just one run?* One run can be lucky or unlucky. With 3 seeds we can see whether a difference is bigger than the random spread. Here the smallest gap is 5.6 points and the largest SD is 1.9, so the gap is real.

### Concept: logarithmic axis

- What it is: an axis where equal steps mean equal *ratios*, not equal differences.
- Where in our code: notebook 05, cell 8.
- Likely question and my answer: *Why a log x-axis?* The training sizes roughly double each step (945, 1,890, 4,725, 9,450, 18,900). On a normal axis the small sizes would be squeezed together. On a log axis they are evenly spaced and easy to compare.

### Concept: Grad-CAM (and its limit here)

- What it is: a way to colour the parts of a picture that pushed the model towards its answer. It takes the feature maps of the last convolution block, weights each one by how much its gradient matters for the predicted class, adds them up and keeps only the positive part.
- Where in our code: `src/gradcam.py` (Rohan), used in notebook 04 (his sections 4 and 5, my check in `src/gradcam_sanity_check.py`).
- Likely question and my answer: *Does Grad-CAM prove what the model looks at?* No. For a 64 x 64 image the last block is only 2 x 2 cells, so the map is a stretched 2 x 2 grid. In our check, hiding the hottest 25% cost 0.064 of confidence and hiding the coldest 25% cost 0.061, so the map did not pick out a more important part. In 160 of 200 images the hottest quarter was the bottom-right and in none the top-left, so the maps mostly show a position, not the content. We show it as a picture, not as proof.

### Concept: confusion matrix, precision and recall

- What it is: a table with one row per real class and one column per predicted class. Recall = of the real images of a class, how many we found. Precision = of the images we called that class, how many really were.
- Where in our code: `src/evaluate.py` (Rohan), read in notebook 04 (his sections 1 to 3, my section 9).
- Likely question and my answer: *Which class is hardest and why?* PermanentCrop has the lowest recall (0.960) and Pasture the lowest precision (0.951). 52% of all mistakes are between the four crop and vegetation classes, whose tiles look alike at 64 x 64 pixels.

## 4. What I understood from the previous part

Baton 1 starts the relay, so there is no previous part to read.

**Baton 5 (2026-11-03), Nitin's answers (recorded word for word from his choices, written down by the AI):**

- Nitin confirmed he understands that Tejash's helper keeps the epoch with the best validation accuracy.
- Nitin did NOT confirm yet: that Tejash's CNN has 3 blocks of Conv, BatchNorm, ReLU, MaxPool; that the test set is scored only once, after training; that the baseline averages about 92.7% over 3 seeds.
- Still TODO: Nitin reads Tejash's notebook 02 and adds the rest in his own words.

## 5. Sources I used

- **Claude Code (AI):** wrote `src/utils.py`, ran the self-check, and drafted the notes in section 2. I still have to read every line and mark anything I do not understand. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
- **Claude Code (AI), 2026-11-03:** wrote `src/train.py`, tested it, and drafted the notes in section 2. It built on Tejash's notebook helper (`notebooks/02_baseline_cnn.ipynb`) and Srivalli's loader (`src/dataset.py`). I still have to read every line. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
- **Claude Code (AI), 2026-11-05:** checked Tejash's checkpoints and Rohan's code, reviewed and fixed Rohan's PR 12, added section 9 to notebook 04, updated the README, drafted `report/part_e_error_analysis_and_explainability_draft.md` and the notes above. I still have to read all of it, and Rohan has to confirm the Part E draft and section 9. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
- **To read:** the PyTorch documentation page "Reproducibility" (it explains `cudnn.deterministic` and why results can differ between machines). Not read yet.

## 6. My 6 viva questions

Six questions a teacher could ask about my part so far (more will be added when I write `train()` in baton 5). At the end, copy them into [docs/viva.md](../viva.md).

1. Question: Why do we set a seed?
   Answer: So a run can be repeated and two settings can be compared fairly. We also run several seeds, because one seed can be lucky.
2. Question: Does the same seed give exactly the same result on every laptop?
   Answer: Not always. Different GPUs and driver versions round decimals a little differently. We fix the seed, record the versions, and report mean and spread.
3. Question: Why do we seed Python, numpy and PyTorch separately?
   Answer: Each one has its own random-number generator. Python's is used when picking items, numpy's by scikit-learn and pandas, and PyTorch's for weights, dropout, shuffling and augmentation.
4. Question: What do `cudnn.deterministic = True` and `cudnn.benchmark = False` do?
   Answer: They make the GPU use the same convolution method every time instead of picking the fastest one, so runs are repeatable. The cost is that it can be a little slower.
5. Question: What does `get_device()` do on a laptop without a GPU?
   Answer: It returns the CPU, so the same code runs everywhere, only slower.
6. Question: Why do we keep folder paths in `utils.py` instead of typing them where we need them?
   Answer: One place to change, no typing mistakes, and it works on every laptop because it is calculated from where the file lives.

Questions about `train()` (baton 5):

7. Question: Why does `train()` keep all the images in memory?
   Answer: Opening and decoding the image files every epoch was slow (Tejash's runs took 45 to 90 seconds per epoch). In memory an epoch takes about 5 seconds on my laptop. It is the same maths, and we checked it against the original loader.
8. Question: How do you know the fast version is correct?
   Answer: The pixels equal Srivalli's loader's pixels, every augmented image is one of the 8 allowed turns or flips, the subsets are the same images her code picks, and full runs reproduced Tejash's results (baseline mean 92.79% against 92.72%, same ordering of the ablations).
9. Question: What happens if `train()` is run twice with the same model, fraction and seed?
   Answer: It gives the same numbers, because of the seed, and the second run replaces its row in `metrics.csv` instead of adding a duplicate.
10. Question: Where is the test set used in `train()`, and where is it not?
    Answer: It is scored once at the end, after the best epoch was chosen on validation. Nothing is decided from it.
11. Question: Why macro-F1 as well as accuracy?
    Answer: The classes have different sizes. Macro-F1 gives each class the same weight, so a weak small class cannot hide behind a good accuracy.

Questions about the data-efficiency study:

12. Question: What does the data-efficiency study show?
    Answer: Transfer learning helps most when labels are scarce. With 945 labelled images (5%) MobileNetV2 reaches 93.69%, about one point better than the scratch CNN with all 18,900 images (92.72%). The gap shrinks from 16.6 points at 5% to 5.6 points at 100% but does not close.
13. Question: Why compare against MobileNetV2 and not ResNet-18?
    Answer: It was the best transfer model on the validation set (98.22% against 97.69%), and the test numbers agree. We chose on validation, not on test.
14. Question: What are the limits of this study?
    Answer: All sizes get the same budget of up to 20 epochs, so small sets get fewer weight updates and may be under-trained. MobileNetV2 was chosen on the full data and not tuned per size. It uses only 3 seeds (seeds 3 and 4 are still missing), and the random split may let neighbouring patches into train and test.

Questions about the Part E checks (notebook 04 section 9 and the Grad-CAM control; drafted by Claude, not checked by Nitin yet):

15. Question: Why does Grad-CAM use the last convolution block, and what is the limit on our data?
    Answer: It is the deepest layer that still keeps where things are, and it holds the most class-specific features. After it the average pooling throws position away. On 64 x 64 images that block is only 2 x 2 cells, so the heat map is very coarse.
16. Question: How did you check that Grad-CAM can be trusted?
    Answer: We painted a patch of 25% of the pixels with the average colour and measured how much the confidence fell. The hottest patch cost 0.064, the coldest 0.061, a random square 0.089. The hottest quarter was the bottom-right in 160 of 200 images and the top-left in none. So the heat maps did not find a more important region; we treat them as pictures, not proof. A first version of the check used scattered random pixels and was unfair, so we redid it with patches.
17. Question: Which classes are confused, and what does the model's confidence tell us?
    Answer: 52% of the 69 mistakes are among AnnualCrop, PermanentCrop, Pasture and HerbaceousVegetation (biggest pair: 7 AnnualCrop called PermanentCrop). Right answers average 99.0% confidence, wrong ones 76.3%, but a "below 80%" rule would flag 60 correct answers to catch 39 of 69 mistakes.

## 7. Open items (not done yet)

Kept up to date by Nitin and the AI. Last updated: 2026-11-05 (Thursday), after PR 12 (Rohan's fixes, merged with Nitin's review fixes) and Tejash's checkpoint check. Gate 2 (results frozen) is Wednesday 21:00, Gate 3 (offline dry run) is Thursday 17:00, the demo is Friday.

| # | Who | What is not done yet | Status |
|---|---|---|---|
| 1 | Rohan | PR 10 (setup), PR 11 and PR 12 (Part E) are merged. He is out of state, so the rest waits until he is available, before the presentation starts and before submission: read the corrections Nitin made (marked "Correction by Nitin" in `results/part_e_analysis.md` and `docs/viva.md`) and section 9 of notebook 04 and the Part E draft in `report/`; his log (`docs/log/rohan.md`): code notes (section 2) are empty, leftover template lines remain, "what I understood" covers only transfer learning (the plan asks about `train.py` and notebooks 02, 03, 05); his AI row says he *will* review every cell, it should say what he checked; his viva answers are one-line definitions and need his own words; three of his scripts call `plt.show()` and hang without a display | Waits for Rohan |
| 2 | Rohan | Seed 4 runs (scratch CNN at 5%, 10%, 25%): done, 3 rows in `results/metrics.csv` from his own CPU script `src/train_part_e.py` (a copy of the training loop, not the shared `train()`). His 5% row reproduced within 0.25 points on my CPU; the shared `train()` on my GPU gives 78.62%, 81.46%, 85.31% for the same seed (not saved), so small-data scores move by a few points. Decide whether to keep his rows or rerun through `train()` | Done, decision open |
| 3 | Srivalli | Seed 3 runs: scratch CNN at 5%, 10%, 25%, on her CPU laptop | Open (train() exists now) |
| 4 | Srivalli | AI row for the Copilot viva answers still says `NOT CHECKED YET`. Her PR 5 checklist was merged with "I can explain every line" unticked and the reviewer boxes unticked | Open |
| 5 | Nitin | The two code questions for Srivalli were never asked: what `make_split` does and why it uses `stratify`; why `get_loader` takes a `fraction` | Open |
| 6 | Tejash | Rerun the baseline and the two ablations through `train()` with seeds 0, 1, 2 (9 runs, about 90 seconds each on Nitin's laptop). His rows in `results/metrics.csv` come from his own notebook helper and are labelled pilot results. His 9 checkpoints were checked on 2026-11-05 (load safely, reproduce his committed test results exactly), so the rerun is now optional | Optional |
| 7 | Tejash | His AI row still says "i have not done anything" in the "what we checked" cell | Open |
| 8 | Rachana | The scratch CNN has **94,986** parameters, not 93,962. The wrong number is in 7 places: her log (2), her viva answers (1), notebook 03 (2) and her report draft (2) | Open |
| 9 | Rachana | If the learning curves are shown in the report: 8 of her 12 histories were rebuilt from log files at lower precision. The result rows themselves are exact | Open (report note) |
| 10 | Everyone | The trained weights are not shared yet. Verified so far on Nitin's laptop: Rachana's 12 checkpoints, Rohan's `mobilenet_v2_full_frac1_seed0.pth` (his results reproduce) and Tejash's 9 scratch-CNN checkpoints (they reproduce his committed numbers exactly). They still exist only in zips and on laptops; Nitin's `models/` folder holds just Rohan's file. Plan: a GitHub Release (not a commit) and `models/README.md` | Open |
| 11 | Nitin | The data-efficiency study (`notebooks/05_data_efficiency.ipynb`): 24 runs, `results/metrics.csv` gained 24 rows, plot in `results/figures/data_efficiency.png`. Merged in [PR 9](https://github.com/Nitinshakthi7/Eyes-in-the-Sky/pull/9). Seeds 3 and 4 will be added when Srivalli and Rohan run theirs | Done (merged) |
| 12 | Nitin | `docs/ai_usage.md` rows for the team plan and for `src/utils.py` still say "Nitin's own review: TODO" | Open |
| 13 | Nitin | In section 1 of this log the three stale links were fixed on 2026-11-05 (PR 2, PR 7 and PR 9) | Done locally, not pushed |
| 14 | Nitin | Section 4 of this log: three statements about Tejash's part are not confirmed yet (3 blocks of Conv, BatchNorm, ReLU, MaxPool; the test set is scored once; the baseline averages about 92.7%) | Open |
| 15 | Nitin | README: "How to run" table, the Results section (all 7 models, data efficiency, Part E summary) and the five GitHub usernames were updated on 2026-11-05 | Done locally, not pushed |
| 16 | Nitin | The 7 questions for the teacher (plan, section 9) have not been sent | Open |
| 17 | Nitin | Report: title page with the contribution table and the repo link on page 1, abstract, data-efficiency results, conclusion, AI-usage paragraph, references, final assembly. Make the repo public and do the fresh-clone test | Open |
| 18 | AI (after Gate 2) | The offline website in `app/` (FastAPI, one page, one predict function). Needs the weights from item 10 | Not started |
| 19 | Everyone | Report caveats to mention: BatchNorm statistics still update in the "frozen" models; the frozen and full comparison also differs in learning rate; a random split can put neighbouring patches in train and test | Open (report note) |
| 20 | Nitin | Housekeeping: `CLAUDE.md` is untracked and not in git; merged branches kept on GitHub: `a-data-split-srivalli`, `c-transfer-rachana`, `d-train-nitin`, `tejash-part-b-complete`; optional branch protection | Optional |
| 21 | Nitin | Notebook 04 section 9, the Part E report draft, the README changes and this log's new notes are on the local branch `e-part-e-completion-nitin`, already merged with the new `main` (167b94d). Nothing is committed or pushed until Nitin says so. Nitin has not checked them yet (AI row says NOT CHECKED YET) | Open: Nitin's check, then commit and pull request |
| 22 | Nitin | Report: the Part E section is a draft written with Claude; Rohan must confirm every number against notebook 04 and the two references (Selvaraju et al. 2017, van der Maaten and Hinton 2008) before it goes in. Also decide whether the report uses Rohan's `results/part_e_analysis.md` or the draft in `report/` (same numbers) | Open (waits for Rohan) |
| 23 | Nitin | Report format and assembly: the plan says Google Docs exported to PDF; item 16 (question 6 to the teacher) is still unanswered | Open |
| 24 | Nitin | I pushed a fix commit onto Rohan's branch `part-e-rohan` (PR 12) and merged it, as Nitin asked. Tell Rohan to `git pull` before he pushes again | Done (tell Rohan) |
