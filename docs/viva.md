# Viva preparation

Every member must be able to explain the whole pipeline, not only their own part.

## Part A - Data and EDA (Srivalli)

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

## Part B - Scratch CNN and ablations (Tejash)

1. Question: Why does the scratch CNN use three convolution blocks with more filters in each block?
   Answer: Tejash to write after practicing.
2. Question: What does BatchNorm do, and how did you test its effect?
   Answer: Tejash to write after practicing.
3. Question: Why use ReLU after each convolution?
   Answer: Tejash to write after practicing.
4. Question: What does MaxPool do in each block?
   Answer: Tejash to write after practicing.
5. Question: Why use global average pooling before the final linear layer?
   Answer: Tejash to write after practicing.
6. Question: How can you tell whether the CNN is overfitting from its learning curves?
   Answer: Tejash to write after practicing.

## Part C - Transfer learning (Rachana)

1. Question: What is transfer learning and why did we use it for EuroSAT satellite images?
   Answer: Transfer learning starts from a deep CNN (ResNet-18 or MobileNetV2) already pretrained on 1.28 million ImageNet photos instead of random weights. We used it because early CNN layers already know how to detect edges, colour contrasts and textures, letting us test whether adapting a deep pretrained network beats training our small `93,962`-parameter scratch CNN from random initialisation.
2. Question: What is the difference between our frozen-backbone and full fine-tuning configurations?
   Answer: In the frozen configurations (`resnet18_frozen`, `mobilenet_v2_frozen`), we set `requires_grad = False` on the pretrained backbone so only the new 10-class linear head trains (`5,130` and `12,810` parameters) at `lr = 0.001`. In the full fine-tuning configurations (`resnet18_full`, `mobilenet_v2_full`), all layers stay trainable (`11.18M` and `2.24M` parameters) and update together at a 10x smaller learning rate (`lr = 0.0001`) so we do not destroy the pretrained features.
3. Question: How did you modify ResNet-18 and MobileNetV2 in `src/models.py` to output 10 EuroSAT classes?
   Answer: Both pretrained models originally output 1,000 ImageNet classes. In `build_transfer_model()`, after optionally freezing the backbone, I replaced ResNet-18's final layer `model.fc` with `nn.Linear(512, 10)` and replaced MobileNetV2's final layer `model.classifier[1]` with `nn.Linear(1280, 10)`. Because a new `nn.Linear` layer defaults to `requires_grad = True`, the new 10-class head is always trainable.
4. Question: Why did the frozen models perform worse than both the fully fine-tuned models and the scratch CNN?
   Answer: `resnet18_frozen` (`86.85% ± 0.32%`) and `mobilenet_v2_frozen` (`87.70% ± 0.29%`) trailed the scratch CNN (`92.72% ± 0.45%`), while `resnet18_full` (`97.98% ± 0.14%`) and `mobilenet_v2_full` (`98.27% ± 0.12%`) improved accuracy by `+11.14 pp` and `+10.58 pp` over their frozen versions. There is a domain shift between ground-level `224x224` ImageNet object photos and top-down `64x64` satellite patches; keeping the deeper convolutional filters fixed limits how well the features separate overhead land-use classes, whereas full fine-tuning adapts all convolutional filters to EuroSAT (noting that frozen and full also use different learning rates: `0.001` vs `0.0001`).
5. Question: What does BatchNorm do, and why do its running statistics still update when `freeze=True`?
   Answer: BatchNorm normalises each channel's activations using the batch mean and variance and applies learnable scale and shift parameters. Setting `param.requires_grad = False` freezes the learnable scale and shift weights, but `running_mean` and `running_var` are stored as buffers, not parameters. Because `src/train.py` calls `model.train()` on the whole network every epoch, those BatchNorm running buffers continue updating on EuroSAT batches even in frozen mode.
6. Question: What did the three-seed results show, and why did you use sample standard deviation (`ddof=1`)?
   Answer: Running seeds 0, 1 and 2 showed very small spread (`0.12%` to `0.32%` SD for the transfer models), meaning training is stable across different random head initialisations, batch shuffle orders and augmentation draws on our fixed split. However, the `+0.29 pp` gap between `mobilenet_v2_full` (`98.27%`) and `resnet18_full` (`97.98%`) is too small across 3 seeds to claim one architecture is statistically superior. We used sample standard deviation (`ddof=1`, dividing by $n - 1$) because 3 seeds are a small sample and it matches Part B's baseline formula.

## Part D - Shared training, data-efficiency study and repo (Nitin)
*Drafted by Claude Code from Nitin's code and results. Nitin has not checked or rewritten these yet. Questions 15 to 17 cover the Part E checks (notebook 04 section 9 and the Grad-CAM control), which are Rohan's part; Rohan writes his own for Part E.*

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
12. Question: What does the data-efficiency study show?
   Answer: Transfer learning helps most when labels are scarce. With 945 labelled images (5%) MobileNetV2 reaches 93.69%, about one point better than the scratch CNN with all 18,900 images (92.72%). The gap shrinks from 16.6 points at 5% to 5.6 points at 100% but does not close.
13. Question: Why compare against MobileNetV2 and not ResNet-18?
   Answer: It was the best transfer model on the validation set (98.22% against 97.69%), and the test numbers agree. We chose on validation, not on test.
14. Question: What are the limits of this study?
   Answer: All sizes get the same budget of up to 20 epochs, so small sets get fewer weight updates and may be under-trained. MobileNetV2 was chosen on the full data and not tuned per size. It uses only 3 seeds (seeds 3 and 4 are still missing), and the random split may let neighbouring patches into train and test.
15. Question: Why does Grad-CAM use the last convolution block, and what is the limit on our data?
   Answer: It is the deepest layer that still keeps where things are, and it holds the most class-specific features. After it the average pooling throws position away. On 64 x 64 images that block is only 2 x 2 cells, so the heat map is very coarse.
16. Question: How did you check that Grad-CAM can be trusted?
   Answer: We painted a patch of 25% of the pixels with the average colour and measured how much the confidence fell. The hottest patch cost 0.064, the coldest 0.061, a random square 0.089. The hottest quarter was the bottom-right in 160 of 200 images and the top-left in none. So the heat maps did not find a more important region; we treat them as pictures, not proof. A first version of the check used scattered random pixels and was unfair, so we redid it with patches.
17. Question: Which classes are confused, and what does the model's confidence tell us?
   Answer: 52% of the 69 mistakes are among AnnualCrop, PermanentCrop, Pasture and HerbaceousVegetation (biggest pair: 7 AnnualCrop called PermanentCrop). Right answers average 99.0% confidence, wrong ones 76.3%, but a "below 80%" rule would flag 60 correct answers to catch 39 of 69 mistakes.

## Part E - Evaluation, Grad-CAM and t-SNE (Rohan)
1. Question: What is a confusion matrix?
   Answer: A confusion matrix shows how many images were correctly and incorrectly predicted for each class. It helps us identify which classes the model confuses.

2. Question: What is Macro-F1, and why did we use it?
   Answer: Macro-F1 calculates the F1 score for each class and takes their average. Every class gets equal importance, even if some classes have fewer images.

3. Question: What is Grad-CAM, and how did we use it?
   Answer: Grad-CAM highlights image regions that influence the model's prediction. We used it to understand which areas of satellite images the model focuses on. [Correction by Nitin, made with Claude on 2026-11-05; Rohan to confirm: on our model the heat maps are only 2 x 2 cells and were hottest in the bottom-right for 80% of 200 test images, and hiding the hottest area hurt no more than hiding the coldest, so Grad-CAM here is a rough picture, not proof of what the model focuses on.]

4. Question: What is our confidence-drop experiment?
   Answer: We masked the top 20% important pixels identified by Grad-CAM in 50 correctly classified images. We compared the original confidence with the confidence after masking. The average confidence dropped by 1.08 percentage points. [Correction by Nitin, made with Claude on 2026-11-05; Rohan to confirm: that experiment had no control. With a control (hide the coldest area, or a random square) the drops were 0.061 and 0.089 against 0.064 for the hottest area, so it does not show that the highlighted pixels matter.]

5. Question: What is t-SNE, and why did we use it?
   Answer: t-SNE reduces high-dimensional features into two dimensions. We used it to visualize similarities and clusters between 1,000 satellite images.

6. Question: Why did we perform error analysis?
   Answer: Error analysis helps us understand model mistakes. We found 69 misclassified images and observed that similar agricultural classes were often confused.
