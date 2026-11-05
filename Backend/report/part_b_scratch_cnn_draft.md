# Scratch CNN methodology and ablation draft

## Methodology

We built a convolutional neural network from scratch for 64 x 64 RGB EuroSAT patches and 10 land-use classes. The baseline has three convolution blocks. Each block uses a 3 x 3 convolution, batch normalisation, ReLU, and 2 x 2 max pooling. The number of filters grows from 32 to 64 to 128. Adaptive average pooling reduces each feature map to one value, and a linear layer produces the 10 class scores.

```text
RGB image (3 x 64 x 64)
  -> Conv(32) -> BatchNorm -> ReLU -> MaxPool
  -> Conv(64) -> BatchNorm -> ReLU -> MaxPool
  -> Conv(128) -> BatchNorm -> ReLU -> MaxPool
  -> Global average pool -> Linear(10) -> class scores
```

All models use the fixed team split and the same training images, ImageNet normalisation, and training-only right-angle rotations and flips. We train with Adam at learning rate 0.001, cross-entropy loss, batches of 64, and at most 20 epochs. The epoch with the best validation accuracy is kept. The test split is scored only after a run is complete.

## Ablations

We change one design choice at a time and repeat each setting with seeds 0, 1, and 2:

| Setting | Change from baseline |
|---|---|
| Baseline | Three blocks; BatchNorm on; dropout 0 |
| No BatchNorm | Remove BatchNorm from all three blocks |
| Two blocks | Use 32 and 64 filters; remove the third block |

## Results and interpretation

The seed 0 baseline completed 20 epochs on the RTX 3050. Its best validation accuracy was **93.01% at epoch 20**; final-epoch training accuracy was **92.88%**. Training loss fell from 0.911 to 0.203, while training accuracy rose from 69.6% to 92.9%. Validation accuracy fluctuated from epoch to epoch but rose from 77.7% at epoch 1 to 93.0% at epoch 20; validation loss ended at 0.210. Both sets of curves improve overall, with no clear train-validation gap at the final epoch. The saved curve is `results/figures/scratch_cnn_learning_curves.png`.

The baseline learned well above the 10% chance level. Validation accuracy varied between epochs, so selecting by validation accuracy gives a more reliable checkpoint than taking an arbitrary epoch.

The table reports test accuracy and macro-F1 as mean ± sample standard deviation over seeds 0, 1, and 2. Each run selected its checkpoint using validation accuracy, then evaluated the test split once. Results are also saved in `results/metrics.csv`; the comparison plot is `results/figures/scratch_cnn_ablation_comparison.png`.

| Setting | Test accuracy (mean ± std) | Test macro-F1 (mean ± std) |
|---|---:|---:|
| Baseline | 92.72% ± 0.45% | 92.48% ± 0.51% |
| No BatchNorm | 88.95% ± 0.63% | 88.45% ± 0.64% |
| Two blocks | 85.31% ± 1.43% | 84.18% ± 1.78% |

Removing BatchNorm reduced mean test accuracy by 3.76 percentage points and macro-F1 by 4.03 points compared with the baseline. The three runs were fairly consistent, so this result supports BatchNorm as useful for this training setup; it does not prove it will help every CNN or dataset. The two-block model reduced mean accuracy by 7.41 points and macro-F1 by 8.30 points. Its larger seed-to-seed spread and lower scores suggest the removed third block supplied useful features, although these results come from only three seeds. The baseline had the best mean score on both metrics.

The notebook used its local, simple training helper because Nitin's shared `src/train.py` still contains a TODO. The helper uses the same fixed team data splits, optimizer, loss, batch size, epoch limit, and validation-based checkpoint selection. For reliable runs on this Windows machine, its training loader uses zero worker processes; the one-worker loader hit a shared-memory allocation error. This is a recorded implementation deviation from the plan's shared-trainer request.

For the learning curves, compare training and validation accuracy and loss. A widening gap, with training performance improving while validation performance stalls or falls, is evidence of overfitting. In the baseline curves both sets improve overall and end close together, so there is no clear large final train-validation gap.
