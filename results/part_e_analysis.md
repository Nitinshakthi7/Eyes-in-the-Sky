Results and Analysis

We should create a Part E report explaining your results in simple research-paper language.



It will contain:

1\. Model Evaluation

&#x20;  - Test accuracy: 98.30%

&#x20;  - Macro-F1: 98.20%

&#x20;  - Correct predictions: 3981/4050

2\. Confusion Matrix Analysis

&#x20;  - Which classes are correctly classified.

&#x20;  - Which classes are confused with each other.

3\. Misclassification Analysis

&#x20;  - Why AnnualCrop, PermanentCrop, and Pasture are confused.

&#x20;  - Examples of incorrectly classified satellite images.

4\. Grad-CAM Explainability

&#x20;  - What the heatmaps show.

&#x20;  - Which image regions influenced predictions.

5\. t-SNE Visualization

&#x20;  - How the 10 classes form clusters.

&#x20;  - Which classes have overlapping features.

6\. Conclusion

&#x20;  - What MobileNetV2 learned.

&#x20;  - Limitations and possible improvements.
## 7. Visual Results



\### 7.1 Confusion Matrix



!\[Confusion Matrix](figures/mobilenet\_v2\_full\_frac1\_seed0\_confusion\_matrix.png)



\### 7.2 Misclassified Images



!\[Misclassified Images](misclassified\_images\_grid.png)



\### 7.3 Grad-CAM Explainability



!\[Grad-CAM](figures/gradcam\_examples.png)



\### 7.4 t-SNE Feature Visualization



!\[t-SNE](figures/mobilenet\_v2\_tsne.png)

