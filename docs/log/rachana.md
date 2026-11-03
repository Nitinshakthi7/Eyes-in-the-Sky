# Work log: Rachana (Part C - Transfer learning (ResNet-18, MobileNetV2))

**Baton:** 4
**Files I own:** `src/models.py` (transfer models), `notebooks/03_transfer_learning.ipynb`
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
| 2026-11-02 | Prepared laptop and environment for transfer-learning task: installed and verified Python, Git, PyTorch, torchvision and project dependencies; verified PyTorch detects NVIDIA RTX 4050 GPU via CUDA; downloaded and verified EuroSAT dataset, ResNet-18 weights and MobileNetV2 weights. | Guided by ChatGPT for setup commands, which I copied, pasted and ran in the terminal myself, sharing outputs for validation. Ran `python check_setup.py --name Rachana --download`. Configured Git global identity, created branch `setup/rachana`, regenerated setup report and staged it using Antigravity. | (pending PR) | Still learning Git and its commands. Understand that CUDA allows PyTorch to use GPU to accelerate training, but not yet completely clear on why CUDA is specifically needed and how it works internally. |

## 2. Code notes (line by line)

For every function or block you write, copy the code here and explain it. Add a new block for each function.

### File: `...` / Function: `...`

```python
# paste the code (or the key lines) here
```

| Line(s) | What it does (simple words) | Concept | Why we used it (and what else we could have used) |
|---|---|---|---|
| | | | |

*Example of a good note (delete this line):* line 12, `nn.BatchNorm2d(32)` | Rescales the 32 feature maps so each has mean about 0 and spread about 1 for every batch | Batch normalisation | Makes training faster and steadier. We could train without it; ablation 1 tests exactly that.

## 3. Concepts I learned

For each concept: (a) what it is in one sentence, (b) where it is in our code, (c) one question a teacher could ask, and my answer.

### Concept: CUDA GPU Acceleration in PyTorch

- What it is: CUDA is NVIDIA's platform that allows PyTorch to run tensor computations on the NVIDIA GPU rather than the CPU, accelerating model training.
- Where in our code: Verified during setup in `check_setup.py` (via `torch.cuda.is_available()`).
- Likely question and my answer: Why do we use CUDA/GPU instead of CPU for training CNNs? Answer: GPUs have thousands of parallel cores suited for matrix operations in deep learning, significantly reducing training time per epoch. (Note: I am still learning why CUDA is specifically needed and how it works internally).

## 4. What I understood from the previous part

Write this BEFORE you start your baton (3 lines is enough). It must be about the part you pulled, not your own.

- (Pending: to be written before starting Baton 4 after Baton 3)

## 5. Sources I used

Papers, documentation pages, videos, and AI chats (say which tool and what it helped with). Nitin uses this for the report's references and for the AI-usage paragraph.

- ChatGPT: Provided terminal commands for installation, configuration and verification; helped troubleshoot setup.
- Antigravity (Gemini 3.8 Flash): Assisted with Git global identity configuration, creating branch `setup/rachana`, regenerating and inspecting setup report, and checking daily log and AI disclosure requirements.
- Project repository: Setup instructions in `README.md` and `docs/plan.md`.

## 6. My 6 viva questions

Six questions a teacher could ask about my part, each with a short answer. At the end, copy them into [docs/viva.md](../viva.md).

1. Question: ...
   Answer: ...
