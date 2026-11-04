# Part C Report Contribution: Introduction, Related Work, and Transfer Learning Results

**Author:** Rachana (`Part C / Baton 4`) — Group 5 (*"Eyes in the Sky: Land-Use Classification from Satellite Images"*)

---

## 1. Introduction

Earth observation satellites continuously capture overhead imagery of the Earth's surface, supporting **Land-Use and Land-Cover (LULC) classification**—the task of assigning semantic categories such as cropland, forest, residential areas, industrial zones, roads, or water bodies to geographic image patches. Automated LULC mapping from satellite imagery is useful for agricultural monitoring, urban-growth tracking, deforestation detection, disaster response, and environmental planning.

While simple spectral indices can separate broad surface types such as dense vegetation and water, many land-use classes share similar colour distributions and differ mainly in spatial arrangement and multi-scale texture (for example, residential versus industrial areas, or rivers versus highways). **Convolutional Neural Networks (CNNs)** address this by learning hierarchical spatial features directly from image pixels. In this project, we use the **EuroSAT RGB** benchmark (Helber et al., 2019), which provides 27,000 labelled `64 × 64` Sentinel-2 satellite patches across 10 land-use classes (detailed in the Dataset section).

Training deep CNNs from random initialisation requires substantial data and compute, whereas compact custom CNNs have limited representational capacity. **Transfer learning** reuses deep backbones—here **ResNet-18** and **MobileNetV2**—pretrained on ImageNet-1K natural photographs (`~1.28M` images, 1,000 classes). A central question of our study is how well ImageNet-pretrained features transfer across the domain shift from ground-level object photos (`224 × 224`) to top-down `64 × 64` satellite patches, and how **frozen feature extraction** compares against **full fine-tuning** and against our custom CNN trained from scratch (`scratch_cnn`).

---

## 2. Related Work

1. **Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).** *"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification."* **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)**, 12(7), 2217–2226. DOI: https://doi.org/10.1109/JSTARS.2019.2918242  
   Introduced the 27,000-image, 10-class EuroSAT Sentinel-2 benchmark and showed that fine-tuning ImageNet-pretrained CNNs on RGB bands achieves strong LULC classification accuracy. This paper provides our benchmark dataset and motivates our transfer-learning baseline.

2. **Nogueira, K., Penatti, O. A. B., & dos Santos, J. A. (2017).** *"Towards better exploiting convolutional neural networks for remote sensing scene classification."* **Pattern Recognition**, 61, 539–556. DOI: https://doi.org/10.1016/j.patcog.2016.07.001  
   Compared training CNNs from scratch, using pretrained networks as fixed feature extractors, and fine-tuning pretrained CNNs on remote-sensing datasets, finding that end-to-end fine-tuning best overcomes the domain gap from natural images. Our Part C experiments directly test this comparison on EuroSAT RGB.

3. **Cheng, G., Han, J., & Lu, X. (2017).** *"Remote Sensing Image Scene Classification: Benchmark and State of the Art."* **Proceedings of the IEEE**, 105(10), 1865–1883. DOI: https://doi.org/10.1109/JPROC.2017.2675998  
   Surveyed CNN-based remote-sensing scene classification and highlighted key overhead challenges including rotation invariance, within-class diversity, and visual similarity across land-use categories. These properties motivate our orientation-invariant training augmentations ($90^\circ$ rotations and flips) and multi-scale CNN comparisons.

4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *"Deep Residual Learning for Image Recognition."* **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, 770–778. DOI: https://doi.org/10.1109/CVPR.2016.90  
   Introduced residual networks with identity shortcut connections that allow gradients to propagate effectively through deeper CNNs. We use ImageNet-pretrained **ResNet-18** (`11.18M` parameters) as our higher-capacity transfer-learning backbone.

5. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).** *"MobileNetV2: Inverted Residuals and Linear Bottlenecks."* **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, 4510–4520. DOI: https://doi.org/10.1109/CVPR.2018.00474  
   Proposed an efficient CNN architecture using depthwise separable convolutions and inverted residual blocks with linear bottlenecks. We evaluate ImageNet-pretrained **MobileNetV2** (`2.24M` parameters, roughly `5×` smaller than ResNet-18) to test whether a lightweight backbone can match ResNet-18 on EuroSAT.

---

## 3. Transfer Learning Results and Discussion

### 3.1 Transfer-Learning Approach and Configurations
In `src/models.py` (`build_transfer_model`), we adapted ImageNet-pretrained **ResNet-18** (`ResNet18_Weights.DEFAULT`, `11,181,642` total parameters) and **MobileNetV2** (`MobileNet_V2_Weights.DEFAULT`, `2,236,682` total parameters) for 10 EuroSAT classes by replacing ResNet-18's final layer `model.fc` with `nn.Linear(512, 10)` (`5,130` parameters) and MobileNetV2's `model.classifier[1]` with `nn.Linear(1280, 10)` (`12,810` parameters). Using the shared training pipeline (`src/train.py`) on the fixed `70/15/15` split (`data/splits.csv`, ImageNet normalisation, training rotations/flips, Adam, batch size `64`, `20` epochs, best validation checkpoint scored once on the `4,050`-image test set), we evaluated four configurations across three seeds (`0, 1, 2`):

- **Frozen feature extraction (`resnet18_frozen`, `mobilenet_v2_frozen`):** Sets `requires_grad = False` on all pretrained backbone parameters so only the new 10-class linear head is updated by Adam (`5,130` and `12,810` trainable parameters) at learning rate `lr = 0.001`. Because `src/train.py` calls `model.train()` on the full model each epoch, `BatchNorm2d` learnable weights ($\gamma, \beta$) remain frozen, but non-parameter `running_mean` and `running_var` buffers continue updating on EuroSAT batches.
- **Full fine-tuning (`resnet18_full`, `mobilenet_v2_full`):** Leaves all layers trainable (`11.18M` and `2.24M` trainable parameters) at a `10×` smaller learning rate (`lr = 0.0001`) to adapt pretrained filters without destabilising them.

### 3.2 Three-Seed Results (`ddof=1`)
Table 1 summarises the three-seed test performance (**mean ± sample standard deviation, `ddof=1`**) from `results/metrics.csv` alongside Tejash's Part B `scratch_cnn` baseline (`93,962` parameters, `lr = 0.001`). The side-by-side comparison chart is saved in `results/figures/transfer_vs_scratch_comparison.png`, and the seed-0 learning curves are saved in `results/figures/transfer_learning_curves_seed0.png`.

| Model | Total / Trainable Params | Learning Rate | Test accuracy (mean ± sample SD) | Test macro-F1 (mean ± sample SD) | Accuracy $\Delta$ vs. `scratch_cnn` |
|---|---:|---:|---:|---:|---:|
| `scratch_cnn` | 93,962 / 93,962 | 0.001 | 92.72% ± 0.45% | 92.48% ± 0.51% | Baseline |
| `resnet18_frozen` | 11,181,642 / 5,130 | 0.001 | 86.85% ± 0.32% | 86.35% ± 0.25% | -5.87 pp |
| `resnet18_full` | 11,181,642 / 11,181,642 | 0.0001 | 97.98% ± 0.14% | 97.91% ± 0.14% | +5.27 pp |
| `mobilenet_v2_frozen` | 2,236,682 / 12,810 | 0.001 | 87.70% ± 0.29% | 87.27% ± 0.31% | -5.02 pp |
| `mobilenet_v2_full` | 2,236,682 / 2,236,682 | 0.0001 | 98.27% ± 0.12% | 98.19% ± 0.13% | +5.56 pp |

### 3.3 Interpretation and Limitations
1. **Frozen vs. full fine-tuning configurations:** `resnet18_full` improves over `resnet18_frozen` by **+11.14 percentage points (pp)** in test accuracy (`+11.56 pp` in macro-F1), and `mobilenet_v2_full` improves over `mobilenet_v2_frozen` by **+10.58 pp** in accuracy (`+10.92 pp` in macro-F1). Because the frozen (`lr = 0.001`) and full (`lr = 0.0001`) configurations change both backbone trainability and learning rate simultaneously, these gains represent the combined difference between the two configurations rather than the isolated effect of unfreezing alone.
2. **Comparison against the scratch CNN baseline:** The linear classification heads trained on frozen ImageNet backbones reach `86.85%` and `87.70%` accuracy (well above the `10%` random baseline), but trail `scratch_cnn` (`92.72%`) by **-5.87 pp** and **-5.02 pp**. A plausible interpretation is that while early ImageNet layers capture generic edge and colour primitives, deeper frozen filters are tuned to ground-level `224 × 224` objects rather than overhead `64 × 64` land-use textures (where stride-32 backbones reduce `64 × 64` inputs to a `2 × 2` feature map before pooling). Conversely, full fine-tuning adapts all convolutional layers to EuroSAT and outperforms `scratch_cnn` by **+5.27 pp** (`resnet18_full`) and **+5.56 pp** (`mobilenet_v2_full`).
3. **ResNet-18 vs. MobileNetV2 and seed stability:** Across seeds `0, 1, 2`, sample standard deviations are small (`0.12%` to `0.32%`), indicating stable convergence across random head initialisations and batch orders on our fixed split. While `mobilenet_v2_full` (`98.27% ± 0.12%`) edges out `resnet18_full` (`97.98% ± 0.14%`) by **+0.29 pp** in mean accuracy (`+0.28 pp` in macro-F1, or roughly `12` test patches out of `4,050`), three seeds on a single fixed split do not establish statistically proven superiority. Rather, both full fine-tuning models achieve comparably strong (`~98%`) accuracy, with MobileNetV2 offering a practical efficiency advantage at roughly `5×` fewer parameters (`2.24M` vs. `11.18M`).
4. **Scope and experimental limitations:** Our findings are bounded by (i) the learning-rate and trainability confounding between frozen and full modes, (ii) `BatchNorm2d` running buffers continuing to update under `model.train()` when `freeze=True`, (iii) evaluating only $n = 3$ seeds on a single random patch split (`data/splits.csv`) using RGB bands rather than all 13 Sentinel-2 bands, (iv) non-identical random augmentation draws between Part B's CPU `torchvision` loader (`02_baseline_cnn.ipynb`) and Part C's GPU tensor augmentation in `src/train.py` (verified by Nitin to match within `92.79%` vs. `92.72%` on `scratch_cnn`), and (v) log-recovered epoch histories (`4` decimal places for loss, `3` for accuracy) for Seeds 1 and 2 in `transfer_all_seeds_histories.json` (while all `results/metrics.csv` test scores and `.pth` checkpoints are exact).
