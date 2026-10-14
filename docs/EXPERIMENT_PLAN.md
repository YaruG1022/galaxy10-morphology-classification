# Experiment plan

All items below are planned. No experimental findings are available.

## Fair comparison and leakage control

- All ten models use the same deduplicated stratified 70/15/15 split.
- Select and record the split seed before creating the split; its value is not yet fixed.
- Keep the held-out test partition closed until the model choices and comparison procedure are frozen.
- Fit feature scaling and principal component analysis (PCA) on training data only. During cross-validation, fit these steps within each training fold.
- Apply augmentation to deep-model training data only.
- Exclude redshift and pixel scale from the main comparison; assess metadata separately.
- Record each run's data version, split identifier, seed, configuration, software environment, and commit.

## Feature plan

Color statistics and band differences in the center and outskirts; concentration, asymmetry, and smoothness; Gini coefficient and M20; image-moment shape measures; and histogram of oriented gradients (HOG) on downsampled grayscale images with PCA reduction.

Feature definitions, regions, preprocessing choices, and retained PCA dimensions remain to be specified by the team.

## Classical models

Multinomial logistic regression, k-nearest neighbors, a support vector machine with a radial basis function kernel, random forest, and XGBoost. Tune with five-fold stratified cross-validation on the training partition. Use class balancing where the model supports it; document alternatives for models without a class-weight option.

## Deep models

- Scratch model: custom CNN with five convolutional blocks.
- Pretrained CNNs: ResNet-50 and ConvNeXt-Tiny.
- Pretrained transformers: ViT-Small/16 and Swin-Tiny.
- Planned input resolution: 224 x 224, resized from 256 x 256.
- Planned training: AdamW, cosine learning-rate schedule, class-weighted cross-entropy, early stopping on validation macro F1, and mixed precision.
- Planned augmentation: rotations and horizontal/vertical flips.
- Scratch normalization uses training-set channel statistics. Pretrained normalization must match the selected pretrained weights; the proposal expects ImageNet normalization.
- Three distinct training seeds per deep model; values and checkpoint identifiers remain to be chosen.

The approximate model-size pairings in the proposal are ResNet-50 versus ViT-Small/16 and ConvNeXt-Tiny versus Swin-Tiny. Size matching alone does not isolate architecture: record differences in pretraining and training recipes when interpreting the comparison.

## Evaluation

Primary metric: macro F1.

Also report accuracy, balanced accuracy, per-class precision/recall/F1, normalized confusion matrices, macro one-vs-rest receiver operating characteristic area under the curve (ROC-AUC), trainable parameters, and training time. Document how valid probabilities or decision scores are obtained for each model.

Report deep-model means and standard deviations over three seeds. Planned figures include class-balance charts, image grids, uniform manifold approximation and projection (UMAP), learning curves, per-class recall comparisons, gradient-weighted class activation mapping (Grad-CAM), attention maps, and error galleries.

### Hypothesis H1

The proposal predicts higher transformer recall for Disturbed and Merging galaxies than for the pretrained CNNs. Report the class-level results and run variation whichever direction the comparison takes. Use the global-attention ViT and local-window Swin results to examine the hypothesis.

The proposal compares performance differences with run-to-run variation. This is a descriptive criterion, not a formal significance test; define the exact aggregation and comparison rule before final evaluation.

### Proposed success targets

- Best fine-tuned model: accuracy at least 0.85 and macro F1 at least 0.80.
- Best deep model: macro F1 at least 0.10 above the best hand-crafted-feature model.
- Best model: recall at least 0.60 for every class.
- Report and explain H1 results with means and standard deviations.
- Every reported number can be reproduced from recorded data, split, configuration, seeds, and code.

These are targets, not promised or observed results.

## Contingencies from the proposal

| Trigger | Planned response |
| --- | --- |
| Best fine-tuned model below 0.75 accuracy or 0.70 macro F1 | Inspect label alignment, normalization, and augmentation; revisit learning rate, epochs, resolution, and gradual unfreezing |
| Deep models do not beat classical features | Inspect small-subset fitting, learning curves, and augmented images before interpreting the result |
| Any class recall below 0.40 | Consider focal loss, oversampling, or targeted augmentation; document remaining limitations |
| Seed variation exceeds between-model differences | Consider five seeds and report an inconclusive comparison when warranted |
| Compute limits | Consider smaller models and shorter runs; any switch to Galaxy MNIST requires an explicit scope decision |
| Duplicate leakage across partitions | Remove duplicates, replace the split, and repeat affected experiments |

## Decisions to resolve before implementation

- Exact duplicate thresholds and match-review procedure.
- Split seed and three training seeds.
- Model checkpoint names, pretrained-data sources, and software versions.
- Hyperparameter search spaces, compute budgets, and stopping rules.
- H1 aggregation and uncertainty reporting.
- A frozen test-evaluation procedure, including any probability calibration.
