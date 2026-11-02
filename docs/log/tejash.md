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
| 2026-11-0_ | | | | |

*Example of a good row (delete this line):* 2026-11-03 | Wrote the scratch CNN | Built 3 blocks of Conv, BatchNorm, ReLU, MaxPool; checked the output shape with a random tensor; trained 3 epochs | PR #4 | Validation accuracy stays at 10%, checking the learning rate

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

### Concept: ...

- What it is:
- Where in our code:
- Likely question and my answer:

## 4. What I understood from the previous part

Write this BEFORE you start your baton (3 lines is enough). It must be about the part you pulled, not your own.

-

## 5. Sources I used

Papers, documentation pages, videos, and AI chats (say which tool and what it helped with). Nitin uses this for the report's references and for the AI-usage paragraph.

-

## 6. My 6 viva questions

Six questions a teacher could ask about my part, each with a short answer. At the end, copy them into [docs/viva.md](../viva.md).

1. Question: ...
   Answer: ...
