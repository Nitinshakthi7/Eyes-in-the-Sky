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

## 4. What I understood from the previous part

Baton 1 starts the relay, so there is no previous part to read.

## 5. Sources I used

- **Claude Code (AI):** wrote `src/utils.py`, ran the self-check, and drafted the notes in section 2. I still have to read every line and mark anything I do not understand. (Also listed in [docs/ai_usage.md](../ai_usage.md).)
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
