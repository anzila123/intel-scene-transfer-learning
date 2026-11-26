# Intel Scene Classification with Transfer Learning (MobileNetV2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anzila123/intel-scene-transfer-learning/blob/main/intel_scene_classification.ipynb)

Classifying natural and urban scenes into 6 categories (**buildings, forest, glacier, mountain, sea, street**) by adapting an ImageNet-pretrained **MobileNetV2**: first as a frozen feature extractor, then with fine-tuning.

Built as part of the Computer Vision module of the **Atomcamp AI/ML Bootcamp (Cohort 19)**.

**Test accuracy: ~91%**, while training only 7% of the model's parameters.

---

## Results

| Metric | Frozen Base | After Fine-Tuning |
|---|---|---|
| Validation Accuracy | 0.9022 | 0.9100 |
| Validation Loss | 0.2691 | 0.2852 |
| Test Accuracy | 0.9122 | 0.9078 |
| Trainable Params | 164,742 | 2,026,182 |

Fine-tuning 12× more parameters gave **no meaningful improvement**. The differences are within run-to-run noise (about ±1 point on 900 test images), and longer fine-tuning led to overfitting. The frozen ImageNet features were already close to what this dataset allows.

<p align="center"><img src="images/confusion_matrix.png" width="520"></p>

**72% of all test errors come from just two class pairs**, glacier ↔ mountain and buildings ↔ street, which were predicted from exploratory analysis before any model was trained.

---

## Pipeline

1. **Data preparation:** pooled the dataset's original train/test folders, sampled 1,000 images per class, and made a stratified 70/15/15 split at the file level (4,200 / 900 / 900), so no image appears in more than one split.
2. **Exploratory analysis:** class distribution, a sample grid, the **average image per class** and **RGB colour histograms**.
3. **Augmentation:** horizontal flip, small rotation (±11°), zoom and brightness. Vertical flip was deliberately excluded because "sky on top" is a real cue in scene photos.
4. **Frozen base:** MobileNetV2 (`include_top=False`, frozen) with a custom head: `GAP → Dropout → Dense(128) → Dropout → Softmax`, trained for 20 epochs.
5. **Fine-tuning:** unfroze the top 54 of 154 layers, used a 100× smaller learning rate (1e-5) with **cosine decay**, and continued from the frozen checkpoint.
6. **Evaluation:** classification report, confusion matrix and visual error analysis.

---

## Highlights

### Exploratory analysis predicted the hardest classes
<p align="center"><img src="images/average_images.png" width="900"></p>

Averaging 200 images per class reveals each class's typical layout. The colour histograms showed that *buildings* and *street* have nearly identical colour profiles, so the model has to rely on structure and shape, which is exactly what pretrained CNN features provide.

### Choosing the architecture
<p align="center"><img src="images/architecture_comparison.png" width="600"></p>

VGG16 has ~40× more parameters than MobileNetV2 for the same ImageNet accuracy. MobileNetV2 was chosen for its small size (lower overfitting risk on 4,200 training images) and fast training on a free Colab GPU.

### Errors are systematic, not random
<p align="center"><img src="images/misclassified.png" width="900"></p>

Misclassifications fall into clear patterns: snowy peaks that are arguably both *glacier* and *mountain* (often predicted with >90% confidence, which suggests label ambiguity), street-level building photos, and **streets without buildings**. The model appears to have learned "street = road + buildings".

### Debugging a silent pipeline bug
The first training run reported a plausible 80% validation accuracy, but training accuracy only moved in steps of exactly 1/32. A partially filled `tf.data` cache meant each epoch trained on **a single batch of 32 images**. I fixed it by removing the cache and adding an assertion on the number of batches per epoch.

---

## Run it yourself

1. Click the **Open in Colab** badge above.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run **Runtime → Run all**. The dataset downloads automatically via `kagglehub` (about 20 minutes in total).

**Stack:** TensorFlow / Keras · MobileNetV2 · scikit-learn · pandas · matplotlib · seaborn

**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle): 17,034 photos of 150×150 pixels across 6 classes.

---

## What I'd try next
- Train on the full dataset (~17k images) to reduce overfitting during fine-tuning
- Grad-CAM to check what regions drive predictions (e.g. buildings vs road for "street")
- Review ambiguous glacier/mountain labels, or treat the problem as multi-label
- Compare EfficientNetB0, and repeat runs with several seeds to measure noise

---

*Developed with AI assistance (Claude) as a learning tutor; see the declaration at the end of the notebook.*

**Author:** Anzila Mansoor
