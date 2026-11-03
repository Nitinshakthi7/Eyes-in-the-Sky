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
6 likely viva questions + short answers - TODO

## Part D - Shared training, data-efficiency study and repo (Nitin)
6 likely viva questions + short answers - TODO

## Part E - Evaluation, Grad-CAM and t-SNE (Rohan)
6 likely viva questions + short answers - TODO
