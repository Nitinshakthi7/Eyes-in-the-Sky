# Work log: Srivalli (Part A - Data, EDA, fixed split, augmentation)

**Baton:** 2
**Files I own:** `src/dataset.py`, `data/splits.csv`, `notebooks/01_eda.ipynb`
**My task card:** [docs/plan.md](../plan.md), section 4, "Baton 2: Part A, Srivalli"

## How to use this file (read once)

- This is YOUR file. Only you edit it. Add to it **every day you work**, even a little, and push it with your pull request.
- Write in simple words, as if you were explaining to a teammate who missed the day.
- Teammates read it to catch up. Nitin reads it to see progress. The teacher may ask any of us about any part, so we must understand each other's work.
- If you do not understand something, write "I do not understand this yet". That is fine. Pretending is not.
- If AI helped you, say so in section 5 and add a row to [docs/ai_usage.md](../ai_usage.md). You still have to explain every line.

---

i changes the spelling of my name for file name and so on

## 1. Daily log

One row per work session. "How" means the steps you really took, not what you planned.

| Date | What I did | How I did it | Commit / PR link | Stuck on / help needed |
|03-11-2026| 
| 2026-11-03 | [written by the AI] Part A, baton 2: set up the laptop, made the fixed split, wrote the data code, ran the EDA notebook, drafted my two report sections | Claude Code did these steps after I gave it the plan and asked it to do my part. Python 3.13 venv with CPU PyTorch; check_setup.py --download; python -m src.dataset; notebook run from a fresh kernel with nbconvert. What I changed from the AI's work: nothing | 2b2600b | the comments made me understand everythign |

```
=== AI SESSION REPORT ===
Date: 2026-11-03
Student: Srivalli (Part A, baton 2)
AI tool and model: Claude Code, Claude Opus 5.5
Topic: Doing the whole of Part A from the task card
What the student asked, in order:
  1. "give the guide files can you do comple shrivalli's part of work as instructed in the files"
  Later in the same chat: add my answers, rename Shrivalli to Srivalli in the repo, commit, push,
  then add my report answers and viva answers and fix their grammar.
What the student ran or checked (word for word): i ran the notebook (reset the kernels and ran it) there were no leakages and content check was ok. the dataset was clearly analysed from its's classes (with smaple images of each class), data augmentation is also done, and many parameters were analysed too include color and other things.
What the AI wrote: src/dataset.py, notebooks/01_eda.ipynb, report/part_a_dataset_and_limitations.md,
  the PR draft, these log entries. It also generated data/splits.csv and results/figures/eda_*.png,
  and check_setup.py rewrote docs/setup/srivalli.md.
What the student wrote or changed: nothing
NOT RUN (the AI did not actually run these):
  - Seed 3 runs (scratch CNN at 5%, 10%, 25%): train() and build_scratch_cnn do not exist yet.
  - get_loader inside a real training loop (only single batches were loaded, including one with num_workers=2).
  - The notebook on Python 3.12 (the team standard). It ran on Python 3.13.14.
  - Restart and Run All in the Jupyter window. It ran from a fresh kernel with nbconvert instead.
  - Any git commit, push or pull request.
```

The AI really ran: the setup check, the split (twice, same file both times), the leakage check, its own tests of the loader and of augment(), and the whole notebook (no errors).

Update, 2026-11-03: after the list above was written, the AI committed `2b2600b` and pushed it to `a-data-split-srivalli`, because Srivalli asked. It did not open a pull request. Srivalli reports running the notebook with a kernel restart in Jupyter.

I am an AI and can make mistakes. Everything marked NOT RUN was not run by me. The student must read and understand every line before submitting it.


*Example of a good row (delete this line):* 2026-11-03 | Wrote the scratch CNN | Built 3 blocks of Conv, BatchNorm, ReLU, MaxPool; checked the output shape with a random tensor; trained 3 epochs | PR #4 | Validation accuracy stays at 10%, checking the learning rate

### Understanding check (2026-11-03)

[written by the AI] The AI asked these two questions after the code (plan section 6A). The answers are Srivalli's, word for word.

**AI's question 1:** Why does `make_split` call `train_test_split` twice instead of making three parts at once, and what would go wrong without `stratify`?

Srivalli's answer:

- Why call it twice?
  `scikit-learn` only splits data into two pieces at a time. To get three pieces (Train/Val/Test), you must split the whole dataset into two, then split one of those pieces a second time.
- Why use `stratify`?
  Without it, random splitting can mess up the class balance. You might get all of Class A in training and all of Class B in testing. Stratification keeps the percentage of each class identical across all three sets.

**AI's question 2:** Why does `augment()` use `RandomRotation((angle, angle))` and not `RandomRotation(90)`?

Srivalli's answer:

- `RandomRotation(90)`
  Rotates the image by a random angle between -90 and +90 degrees.
- `RandomRotation((angle, angle))`
  Forces the model to rotate the image by exactly that one specific angle every single time, turning off the randomness.

## 2. Code notes (line by line)

For every function or block you write, copy the code here and explain it. Add a new block for each function.

[written by the AI] The code itself is in `src/dataset.py`; each note names the lines it explains.

### File: `src/dataset.py` / Function: `load_eurosat`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `datasets.EuroSAT(root=str(root), download=False)` | Opens the images already in `data/eurosat/2750` | torchvision dataset | Gives us `.samples` and `.classes` ready-made. `download=False` so it never downloads by surprise. We could list the folders by hand, but this is less code |

### File: `src/dataset.py` / Function: `make_split`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `for path, label in eurosat.samples` and `relative_to(DATA_DIR).as_posix()` | Makes one row per image and keeps the path from the data folder, with `/` | Portable paths | The full path is different on each laptop, so a full path would break for teammates |
| `train_test_split(df.index, train_size=0.7, stratify=df["label"], random_state=seed)` | Cuts 70% train, 30% rest, with the same class shares in both | Stratified split | Without `stratify`, a class could by chance be under-represented in a part. `random_state` makes it repeatable |
| second `train_test_split(rest_idx, train_size=0.5, stratify=df.loc[rest_idx, "label"], ...)` | Cuts the rest in half: validation and test | Three-way split | Validation is for choosing, test is for the final score. The stratify labels must match `rest_idx` |
| `df.loc[train_idx, "split"] = "train"` (3 lines) | Writes each image's part into a new column | pandas `.loc` | Every image gets exactly one value, so one file cannot be in two parts |
| `df.to_csv(SPLITS_CSV, index=False, lineterminator="\n")` | Saves the split | Reproducibility | The file is committed so everyone shares one split. `lineterminator` stops git seeing fake changes between laptops |

### File: `src/dataset.py` / Function: `check_split`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `assert sizes == SPLIT_SIZES` | Stops if sizes are not 18,900 / 4,050 / 4,050 | `assert` | A wrong split should fail loudly, not silently |
| `assert df["path"].is_unique` | Stops if any file is listed twice | Data leakage | Each file has one row with one split, so unique paths mean no file is in two splits |
| `pd.crosstab(...)` and the `for` loop | Counts images per class per split and compares with the exact share (at most 1 off) | Stratification check | Proves the share is kept. 1 is allowed because image counts must be whole numbers |

### File: `src/dataset.py` / Function: `augment`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `RandomChoice([RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)])` | Picks one of four exact angles | Data augmentation | `(angle, angle)` means from angle to the same angle, so it is exact. Plain `RandomRotation(90)` would allow any angle like 37 degrees, which blurs pixels and leaves empty corners |
| `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)` | Flips with a 50% chance each | Label-safe transforms | With the rotations this gives all 8 orientations of a square (tested: all 8 appeared and pixels stayed exact) |

### File: `src/dataset.py` / Function: `make_transform`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `steps = [augment()] if use_augment else []` | Adds augmentation only if asked | Conditional pipeline | Ablation "augmentation off" and val/test both need it off |
| `ToTensor()`, `Normalize(IMAGENET_MEAN, IMAGENET_STD)` | Turns the image into numbers 0 to 1, then subtracts the mean and divides by the std | Normalisation | Team setting for all models, so the comparison is fair. Augmentation goes first because it works on the PIL image |

### File: `src/dataset.py` / Class: `EuroSATDataset`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `df[df["split"] == split].reset_index(drop=True)` | Keeps only the rows of one part and renumbers them 0, 1, 2... | PyTorch `Dataset` | `__getitem__` asks for row number i, so the numbers must have no gaps |
| `__len__` | Says how many images | `Dataset` rule | The DataLoader needs it to plan batches |
| `Image.open(DATA_DIR / row["path"]).convert("RGB")` | Reads one image from disk when needed | Lazy loading | Saves memory compared with loading 27,000 images at once. `convert("RGB")` makes sure there are 3 channels |
| `return self.transform(image), int(row["label"])` | Gives back (image, label) | `Dataset` rule | This is what training loops expect |

### File: `src/dataset.py` / Function: `get_loader`

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| `if split != "train" and (fraction != 1.0 or augment): raise ValueError` | Refuses to shrink or augment val/test | Guard against mistakes | Val and test must stay the same for every run, so a mistake here should stop the run |
| `train_test_split(data.df.index, train_size=fraction, stratify=data.df["label"], random_state=seed)` | Picks a smaller training set with the same class shares | Stratified subset | A plain random 5% could leave a class with too few images. Different seeds give different subsets, which is what the study measures |
| `sorted(keep)` and `reset_index(drop=True)` | Keeps file order and renumbers | Repeatability | The row order then does not depend on how the split shuffled them |
| `torch.Generator().manual_seed(seed)` | Makes the shuffle order repeatable | Random generator | The same seed gives the same batches (tested) |
| `shuffle=(split == "train")` | Shuffles only training data | Mini-batch training | Shuffling helps training. Val and test do not need it, and a fixed order keeps results easy to match to files |

*Example of a good note (delete this line):* line 12, `nn.BatchNorm2d(32)` | Rescales the 32 feature maps so each has mean about 0 and spread about 1 for every batch | Batch normalisation | Makes training faster and steadier. We could train without it; ablation 1 tests exactly that.

## 3. Concepts I learned

For each concept: (a) what it is in one sentence, (b) where it is in our code, (c) one question a teacher could ask, and my answer.

[written by the AI]

### Concept: Stratified split

- What it is: a split that keeps each class's share the same in every part.
- Where in our code: `make_split` and the `fraction` part of `get_loader`.
- Likely question and short answer: why stratify? Classes are not equal in size (2,000 to 3,000 images), and stratifying stops a part getting too few of a class by chance. It matters most at 5%, where the smallest class has only 70 images.

### Concept: Data leakage

- What it is: information from the test data reaching training, which makes the score look better than it really is.
- Where in our code: `check_split`, and the MD5 check at the top of notebook 01.
- Likely question and short answer: is our split free of leakage? No file and no identical picture is in two splits, which we checked. Neighbouring patches can still be in both, which we did not rule out and which we say in the report.

### Concept: Label-safe augmentation

- What it is: changing a training image in a way that keeps its true label.
- Where in our code: `augment`.
- Likely question and short answer: why only on training images? It gives the model more variety to learn from. Validation and test must stay the same so that scores can be compared fairly.

### Concept: Normalisation

- What it is: shifting and scaling pixel values using a fixed mean and std.
- Where in our code: `make_transform`.
- Likely question and short answer: is the data exactly mean 0 and std 1 after this? No. One batch gave mean -0.161 and std 0.833, because these numbers come from ImageNet. It is a team choice so all models get the same input, and the pretrained models expect it.

## 4. What I understood from the previous part

Write this BEFORE you start your baton (3 lines is enough). It must be about the part you pulled, not your own.

- Nitin set up the file structure and implemented the seed function.

## 5. Sources I used

Papers, documentation pages, videos, and AI chats (say which tool and what it helped with). Nitin uses this for the report's references and for the AI-usage paragraph.

[written by the AI]

- P. Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS, vol. 12, no. 7, pp. 2217-2226, 2019. https://arxiv.org/abs/1709.00029 (the AI read the dataset facts from the paper)
- D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," Ecography, vol. 40, no. 8, pp. 913-929, 2017. doi:10.1111/ecog.02881 (the AI confirmed the details from search results, not the full text)
- EuroSAT repository, licence statement: https://github.com/phelber/EuroSAT
- Function behaviour of scikit-learn, torchvision and pandas: from the AI's own knowledge, then tested by running the code. No specific documentation page was opened.
- AI chat: Claude Code (Claude Opus 5.5), see the AI SESSION REPORT in section 1.
- [written by Claude Code] AI chat: the viva answers in section 6 came from Copilot (Srivalli confirmed; see the note there). Copilot session report: TODO Srivalli.

## 6. My 6 viva questions

Six questions a teacher could ask about my part, each with a short answer. At the end, copy them into [docs/viva.md](../viva.md).

1. Question: What dataset did you use for the EDA?
   Answer: We used EuroSAT: 27,000 RGB satellite images, each 64x64 pixels, grouped into 10 land-use classes.
2. Question: How did you divide the images for training and testing?
   Answer: We used a fixed, stratified split: 70% for training, 15% for validation and 15% for testing. Stratification keeps each class represented in similar proportions.
3. Question: How did you check that the data split had no leakage?
   Answer: We checked that no image appeared in more than one split, and we compared file fingerprints to catch identical images saved under different names.
4. Question: What did you learn about the class distribution?
   Answer: The classes are mildly imbalanced. The largest classes have 3,000 images, while Pasture has 2,000, so we also consider macro-F1, which gives each class equal weight.
5. Question: What did the sample images and colour analysis show?
   Answer: Some classes look different in colour, but others are similar. The samples also show that one patch can contain several land types, which can make its single label imperfect.
6. Question: Why did you augment training images, and which transforms did you use?
   Answer: We used right-angle rotations and horizontal and vertical flips to create varied training examples. These are suitable for satellite images because there is no fixed "up" direction. Validation and test images were left unchanged.

[written by Claude Code] Srivalli pasted the six questions and answers above. Srivalli confirmed they came from Copilot, so they are AI-written. Claude Code only fixed grammar and format. Copilot session report: TODO Srivalli.

### Extra practice questions (from the task card and Claude Code)

[written by the AI] The plan does not let the AI write viva answers, so the questions are here and the answers are yours to write.

1. Question: Why did we use a stratified split?
   Answer: Without it, random splitting can mess up the class balance. You might get all of Class A in training and all of Class B in testing. Stratification keeps the percentage of each class identical across all three sets.
2. Question: What is data leakage, and how do we prove there is none?
   Answer: TODO Srivalli
3. Question: Why are rotations and flips fine for satellite images but not for digits?
   Answer: TODO Srivalli
4. Question: Why is the test set kept untouched?
   Answer: TODO Srivalli
5. Question: Why can a random split still be too optimistic for satellite patches?
   Answer: TODO Srivalli
6. Question: Which classes look hard to separate in the EDA, and what in our plots shows it?
   Answer: TODO Srivalli
