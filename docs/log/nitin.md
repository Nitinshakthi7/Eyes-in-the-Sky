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
| 2026-11-02 | Baton 1: wrote `src/utils.py` (seed, device, paths) | With Claude Code. Wrote 3 small helpers, then a self-check at the bottom of the file. Ran `python -m src.utils`: passed. Wrote the notes below. | (PR for this branch: TODO) | I still have to read every line below and mark what I do not understand yet |
| 2026-11-03 | Baton 5: wrote `src/train.py` (the shared `train()`), earlier than planned because Tejash had already finished | With Claude Code. Read Tejash's pushed branch first and checked his settings against the team settings. Wrote `train()` from his helper, plus a `fraction` option, all images kept in memory, and a results file that replaces a repeated run's row. Tested it on the real data: every check passed, and it reproduced his baseline (mean of 3 seeds 92.79% against his 92.72%) at 4.6 s per epoch. Wrote the notes below. | (branch `d-train-nitin`, not pushed yet) | Nitin's answers: understands it all ("I understand it all"), no changes to the AI's work. He first said "None of these yet", then ran `python -m src.train` himself and it passed; asked to read the code and the notes, he then said: "I have read and understood" (2026-11-03) |

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

## 4. What I understood from the previous part

Baton 1 starts the relay, so there is no previous part to read.

**Baton 5 (2026-11-03), Nitin's answers (recorded word for word from his choices, written down by the AI):**

- Nitin confirmed he understands that Tejash's helper keeps the epoch with the best validation accuracy.
- Nitin did NOT confirm yet: that Tejash's CNN has 3 blocks of Conv, BatchNorm, ReLU, MaxPool; that the test set is scored only once, after training; that the baseline averages about 92.7% over 3 seeds.
- Still TODO: Nitin reads Tejash's notebook 02 and adds the rest in his own words.

## 5. Sources I used

- **Claude Code (AI):** wrote `src/utils.py`, ran the self-check, and drafted the notes in section 2. I still have to read every line and mark anything I do not understand. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
- **Claude Code (AI), 2026-11-03:** wrote `src/train.py`, tested it, and drafted the notes in section 2. It built on Tejash's notebook helper (`notebooks/02_baseline_cnn.ipynb`) and Srivalli's loader (`src/dataset.py`). I still have to read every line. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
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
