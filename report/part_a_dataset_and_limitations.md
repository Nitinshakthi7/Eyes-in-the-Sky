# Report sections from Part A (Srivalli)

Two sections for Nitin to place in the final report. Figures are in `results/figures/`. All numbers come from `notebooks/01_eda.ipynb`. The limitations section does not depend on any model result, and it should be re-read once the final results exist.

---

## Dataset

**Source.** We use the RGB version of EuroSAT [1]: 27,000 labelled 64x64 pixel JPEG patches from the Sentinel-2 satellite, at 10 metres per pixel, in 10 land-use and land-cover classes. The RGB version combines the red, green and blue bands (B04, B03, B02). The authors report that the patches come from over 34 countries, that they chose images with low cloud cover recorded throughout the year, and that they checked all images by hand [1].

**Split.** We made one fixed split and every experiment uses it. `make_split(seed=42)` calls scikit-learn's `train_test_split` twice with `stratify`: first 70% train and 30% rest, then the rest 50/50 into validation and test. The result is stored in `data/splits.csv`, which is committed to git, so all five members use exactly the same images even though the dataset itself is not in git. Stratifying keeps each class's share the same in all three parts.

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| AnnualCrop | 3,000 | 2,100 | 450 | 450 |
| Forest | 3,000 | 2,100 | 450 | 450 |
| HerbaceousVegetation | 3,000 | 2,100 | 450 | 450 |
| Highway | 2,500 | 1,750 | 375 | 375 |
| Industrial | 2,500 | 1,750 | 375 | 375 |
| Pasture | 2,000 | 1,400 | 300 | 300 |
| PermanentCrop | 2,500 | 1,750 | 375 | 375 |
| Residential | 3,000 | 2,100 | 450 | 450 |
| River | 2,500 | 1,750 | 375 | 375 |
| SeaLake | 3,000 | 2,100 | 450 | 450 |
| **Total** | **27,000** | **18,900** | **4,050** | **4,050** |

**Leakage check.** `check_split` stops with an error unless the split sizes are exact, every file appears only once, and every class keeps its share. Because two different file names could hold the same picture, the notebook also compares an MD5 fingerprint of every file's bytes: no identical file exists in more than one split. This excludes exact copies only, not neighbouring patches (see Limitations). The mean colours and histograms below use training images only, so the test set is not used to make any decision.

**Exploratory analysis.** The classes are mildly imbalanced: the largest have 3,000 images and Pasture has 2,000, a ratio of 1.5 to 1 (`eda_class_counts.png`). This is one reason we report macro-F1 as well as accuracy. The sample grid (`eda_samples.png`) shows that Forest and SeaLake are smooth and dark, while Industrial and Residential have much fine bright detail. It also shows that a patch often contains more than its label: Highway patches include fields and buildings.

Mean colour separates some classes but not all (`eda_mean_colour.png`). Forest is darkest (R, G, B = 39, 65, 76) and Industrial brightest (123, 126, 134), but AnnualCrop (130, 120, 117) is close to PermanentCrop (122, 116, 114), and Pasture (62, 88, 94) is close to River (67, 87, 97). The model therefore needs shape and texture as well as colour. The histograms (`eda_histograms.png`) agree: 5% to 95% of Forest's pixel values lie between 33 and 84, against 66 to 243 for Industrial. AnnualCrop and Industrial also show a spike at 255, where bright pixels are clipped to the maximum value.

**Preprocessing and augmentation.** All models get images normalised with the ImageNet mean and standard deviation, so the comparison is fair. These numbers do not match EuroSAT exactly: one normalised training batch had mean -0.161 and standard deviation 0.833. Training images are augmented with a random rotation of 0, 90, 180 or 270 degrees and random horizontal and vertical flips. These are label-safe because a satellite looks straight down, so there is no fixed "up", and they only move pixels (`eda_augment.png`). Validation and test images are never augmented. For the data-efficiency study, `get_loader` takes a stratified share of the training split only. At 5% this leaves 945 images, with 70 in the smallest class. Validation and test never shrink.

---

## Limitations, ethics and future work

**Our random split may be optimistic.** The split is random at the level of single patches. Patches cut from the same satellite scene or the same region can look very alike, so neighbouring patches can land in both train and test. Our checks exclude identical files, not this. Random splits of spatially related data are known to underestimate the true error [2], so our test scores describe new patches from the areas and scenes already seen, and may overstate performance in a new region. We use location-free file names and classes only, and we have not measured the size of this effect.

**Other limitations of the data.**

- *One label per patch.* A 640 m x 640 m patch often holds several land types, as the Highway samples show, so some "errors" may be partly a labelling limit.
- *Geographic scope.* The data covers European countries [1]. Crops, buildings and landscapes differ elsewhere, so we make no claim about performance outside Europe.
- *Resolution and bands.* At 10 m per pixel, small objects are only a few pixels wide. We use 3 of the 13 available bands, and we also give the ImageNet models 64x64 inputs, much smaller than the images they were trained on.
- *Fixed split.* Repeated seeds show the variation from training and from the data subset, not from the choice of test set, because the test set stays the same.

**Ethics.** The imagery is open: the EuroSAT repository states an MIT licence for the dataset and says Sentinel data is free and open to the public, and it asks users to review the Copernicus terms [3]. At 10 m per pixel, individual people and vehicles cannot be identified, so the direct privacy risk is low. The larger risk is how outputs are used. Land-use maps can feed decisions on farming subsidies, planning and disaster response, and those decisions affect people. A model tested only on European patches should not be used for such decisions elsewhere without new validation, and any real use should keep human review. Per-class results matter here too, because a good average can hide a weak class. Finally, we disclose our AI use in `docs/ai_usage.md`.

**Future work.**

- Build a block split that keeps nearby patches in the same part. The authors describe the dataset as georeferenced [1], so location data should be available in the multispectral version. Comparing the two splits would measure the effect described above.
- Use all 13 multispectral bands, the stretch goal in the brief, by adapting the first convolution layer.
- Test on patches from outside Europe to measure how well the model generalises.
- Compute normalisation numbers from the EuroSAT training split and compare them with the ImageNet numbers.

**References for these sections**

[1] P. Helber, B. Bischke, A. Dengel and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019. doi:10.1109/JSTARS.2019.2918242.

[2] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913-929, 2017. doi:10.1111/ecog.02881.

[3] EuroSAT repository, https://github.com/phelber/EuroSAT.
