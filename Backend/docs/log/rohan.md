# Work log: Rohan (Part E - Evaluation, Grad-CAM, t-SNE)

**Baton:** 6
**Files I own:** `src/evaluate.py`, `src/gradcam.py`, `notebooks/04_evaluation_and_explainability.ipynb`
**My task card:** [docs/plan.md](../plan.md), section 4, "Baton 6: Part E, Rohan"

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
| 2026-11-05 | Completed Part E evaluation and explainability experiments | Evaluated MobileNetV2, analyzed 69 errors, generated confused-pair examples, Grad-CAM for 10 classes, confidence-drop experiment, t-SNE on 1,000 images, and scratch CNN seed 4 experiments at 5%, 10%, and 25%. Created Part E report and Notebook 04. | Pending | Final Git submission |

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
### Concept: Confusion Matrix
- What it is: A table showing correct and incorrect predictions for every class.
- Where in our code: `src/evaluate.py`
- Likely question and my answer: A confusion matrix helps identify which classes the model confuses.

### Concept: Macro-F1
- What it is: The average F1 score across all classes, giving equal importance to each class.
- Where in our code: `src/evaluate.py`
- Likely question and my answer: Macro-F1 is useful because it evaluates every class equally.

### Concept: Grad-CAM
- What it is: A visualization showing image regions that influence a CNN prediction.
- Where in our code: `src/gradcam.py`
- Likely question and my answer: Grad-CAM uses gradients and feature maps to highlight important image regions.

### Concept: t-SNE
- What it is: A technique that reduces high-dimensional features into two dimensions for visualization.
- Where in our code: `src/visualize_tsne.py`
- Likely question and my answer: t-SNE helps visualize whether images from different classes form clusters.

### Concept: ...

- What it is:
- Where in our code:
- Likely question and my answer:

## 4. What I understood from the previous part

- Transfer learning uses a pretrained model instead of learning all image features from scratch.
- ResNet18 and MobileNetV2 were compared using frozen and fully fine-tuned models.
- Multiple seeds and test metrics were used to compare model performance.

-

## 5. Sources I used
- OpenAI ChatGPT: Helped explain evaluation metrics, Grad-CAM, t-SNE, and confusion matrices in simple language; assisted with debugging, notebook organization, report drafting, and work-log documentation. I reviewed and ran the code and checked the generated outputs.
- PyTorch documentation: Used for understanding model evaluation and tensor operations.
- scikit-learn documentation: Used for classification metrics and t-SNE concepts.

Papers, documentation pages, videos, and AI chats (say which tool and what it helped with). Nitin uses this for the report's references and for the AI-usage paragraph.

-

## 6. My 6 viva questions

1. Question: What is a confusion matrix?
   Answer: It shows correct and incorrect predictions for each class.

2. Question: What is Macro-F1?
   Answer: It calculates the F1 score for every class and takes their average, giving equal importance to all classes.

3. Question: Why did we use MobileNetV2?
   Answer: It is a lightweight pretrained CNN that provides useful image features and performs well with transfer learning.

4. Question: What is Grad-CAM?
   Answer: It highlights image regions that influence a CNN's prediction using gradients and feature maps.

5. Question: What is t-SNE?
   Answer: It reduces high-dimensional features to two dimensions so we can visualize similarities and class clusters.

6. Question: Why did we perform error analysis?
   Answer: To identify confused classes, understand model mistakes, and find areas for improvement.


Six questions a teacher could ask about my part, each with a short answer. At the end, copy them into [docs/viva.md](../viva.md).

1. Question: ...
   Answer: ...
