# Galaxy Morphology Classification

[中文](README.zh-CN.md)

Use the ten-class Galaxy10 DECaLS image dataset to compare classical classifiers on handcrafted features, a convolutional neural network (CNN) trained from scratch, and pretrained CNNs and vision transformers.

**Status: data preparation, exploratory data analysis (EDA), and handcrafted feature engineering are complete; classifier training and performance evaluation have not started.** The raw file contains 17,675 images, and cleaning retains 17,655. Records, figures, and executed notebooks for the first three modules are saved in this repository.

## Research questions

- How does classification performance change from classical models on handcrafted features to a scratch CNN and fine-tuned pretrained models?
- Do vision transformers achieve higher recall on Disturbed and Merging galaxies than comparably sized CNNs?
- Which classes and images confuse different model families?
- Can class-weighted training improve minority-class performance?

## Planned model comparison

| Family | Models |
| --- | --- |
| Handcrafted features | Multinomial logistic regression; k-nearest neighbors (kNN); radial basis function support vector machine (RBF SVM); random forest; Extreme Gradient Boosting (XGBoost) |
| CNN trained from scratch | Custom network with five convolutional blocks |
| Pretrained CNNs | ResNet-50; ConvNeXt-Tiny |
| Pretrained vision transformers | ViT-Small/16; Swin-Tiny |

A majority-class baseline always predicts the largest class. It is an additional sanity check and is excluded from the ten-model count.

## Completed modules

### 1. Data preparation

Checked the source checksum, image dimensions, data types, and class labels; screened for unusable images and duplicates; and recorded exclusion reasons.

Excluded 20 images: one constant image, 17 redundant cutouts, and two overlapping cutouts with conflicting class labels. The retained dataset contains 17,655 images. The raw file is unchanged, and exclusions are recorded in the sample manifest. Duplicate screening has documented limits and cannot guarantee that every duplicate has been removed.

All models must use the same frozen split:

| Partition | Images | Original-row index file |
| --- | ---: | --- |
| Training | 12,359 | [train_indices.json](metadata/data_v1/train_indices.json) |
| Validation | 2,650 | [validation_indices.json](metadata/data_v1/validation_indices.json) |
| Test | 2,646 | [test_indices.json](metadata/data_v1/test_indices.json) |

The proportions are approximately 70/15/15, with random seed 42. These files contain zero-based row indices into the unchanged raw file. Images are read from that single file rather than copied into three image folders.

See [the data record](docs/DATA.md) for details and [sample_manifest.csv](metadata/data_v1/sample_manifest.csv) for each image's inclusion, exclusion, and partition assignment.

### 2. Exploratory data analysis

Produced seven figures covering class counts, image examples, class-average images, stored-pixel intensity, stored-channel differences, redshift, and pixel scale. All image-derived statistics and gallery selections use training images only; full retained class counts come from the frozen sample manifest. Four examples per class were selected using a fixed random seed.

The largest class contains 2,640 images and the smallest 333, a ratio of about 7.93:1. Overall accuracy can therefore hide poor minority-class performance. Neighboring sources, background, and differences in target size can also affect whole-image statistics.

These figures describe the data and do not establish classification performance. See [the exploratory record](docs/eda/README.md) and the executed walkthrough in [01_exploration.ipynb](notebooks/01_exploration.ipynb).

### 3. Handcrafted feature engineering

Extracted stored-channel statistics, morphology proxies, shape descriptors, quality flags, and Histogram of Oriented Gradients (HOG) from all 17,655 retained images.

Each image has 54 scalar/quality descriptors and 1,764 raw texture features. Imputation, standardization, and Principal Component Analysis (PCA) learn their parameters from training rows only. PCA reduces the texture block to 64 components, giving 118 processed inputs per image.

- Class labels, original row indices, redshift, pixel scale, and sky coordinates are excluded from input features.
- Training rows have zero undefined scalar entries; 2,082 target masks touch the fixed detection boundary, or 16.85% of training images.
- The fixed 64 principal components retain 42.03% of standardized training texture variance. This setting was chosen in advance and has not been evaluated through classification experiments.
- Morphology proxies are derived from display composites, not calibrated astronomical measurements.
- Uniform Manifold Approximation and Projection (UMAP) of training features is used only to inspect a two-dimensional distribution. Labels do not enter fitting, and the projection is not evidence of classification performance. Pretrained-representation UMAP remains pending.

See [the feature record](docs/features/README.md), [fixed configuration](configs/features_v1.json), and [02_handcrafted_features.ipynb](notebooks/02_handcrafted_features.ipynb).

## Evaluation rules

Macro-averaged F1 is the planned primary metric, with per-class recall also reported. Keep the test set closed until model choices are frozen. Each deep model is planned to run with three random seeds; their values remain to be chosen.

For future cross-validation, pass **raw descriptors** to `make_model_pipeline(estimator)` in `galaxy_features.py` and put the entire pipeline inside `GridSearchCV`. Each training fold must refit imputation, scaling, and dimensionality reduction to avoid data leakage. Do not use the already transformed full-training exports for inner cross-validation.

The saved full-training preprocessor is for the eventual final training fit and held-out inference. See [the experiment plan](docs/EXPERIMENT_PLAN.md) for details.

## Run locally

Use Python 3.12. Download `Galaxy10_DECals_NoDuplicated.h5` from [Zenodo](https://zenodo.org/records/10845026) and place it under `data/raw/`. Run the following commands from the repository root.

Check the raw file and saved split:

```powershell
python -m pip install -r requirements-data.txt
python check_data.py
python validate_data_preparation.py
```

`check_data.py` audits the unchanged raw file and will still report its recorded issues. `validate_data_preparation.py` checks cleaning records, the source checksum, and the frozen split, including complete/disjoint membership and separation of known duplicate candidates.

Rebuild exploratory analysis:

```powershell
python -m pip install -r requirements-eda.txt
python eda.py
python validate_eda.py
```

Extract and check handcrafted features:

```powershell
python -m pip install -r requirements-features.txt
python build_features.py
python validate_features.py
```

Run the existing test cases:

```powershell
python -m unittest discover -s tests -v
```

A project virtual environment is prepared on the current machine and can be invoked as `.venv\Scripts\python.exe`. Feature extraction processes every retained image; runtime depends on the machine.

`prepare_data.py` rebuilds screening and preparation and refuses to overwrite a completed frozen version. To review again, use new output/cache directories, run `--scan-only`, record review decisions, and then run `--finalize`; see [the data record](docs/DATA.md). Classifier training commands will be added when that module is implemented.

## File locations

| Path | Contents |
| --- | --- |
| `data/raw/` | Raw image file, kept locally |
| `metadata/data_v1/` | Frozen split, sample manifest, cleaning decisions, and validation records |
| `data/processed/preparation_v1/` | Local thumbnail cache and visual review sheets |
| `docs/eda/` | Exploratory figures, aggregate tables, and interpretations |
| `data/processed/eda_v1/` | Local per-image diagnostics and cached arrays |
| `docs/features/` | Feature methods, diagnostic figures, and limitations |
| `metadata/features_v1/` | Feature definitions, configuration, fit indices, and validation records |
| `data/processed/features_v1/` | Local feature matrices and fitted preprocessor |
| `notebooks/` | Executed notebooks |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Planned methods, metrics, targets, and open decisions |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Remaining stages and planned responsibilities |
| [docs/ENVIRONMENT_PLAN.md](docs/ENVIRONMENT_PLAN.md) | Environment and compute plans |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaboration and experiment-recording conventions |
| [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE) | Task and experiment templates |
| [.gitignore](.gitignore) | Exclusions for local environments, secrets, datasets, model artifacts, and `AGENTS.md` |

Under `metadata/data_v1/`, `review_decisions.json` records exclusions and review provenance; `summary.json` records counts, seed, file hashes, and screening limits.

Under `data/processed/features_v1/`, `raw_features.npz` contains raw descriptors; `train_features.npz`, `validation_features.npz`, and `test_features.npz` contain the processed 118-feature rows for each partition. Labels and original row indices are stored separately from the input array `X`. The scalar feature table's `original_index`, `label`, and `split` columns are also metadata, not model inputs.

Raw data, large matrices, and local environments are excluded by `.gitignore` and are not uploaded with the code. Small configurations, records, and curated figures are kept in version control.

## Team responsibilities

These are planned responsibilities, not a statement that every task is complete:

- **Yaru Gao:** data acquisition, cleaning, duplicate checks, shared split, exploratory analysis, and handcrafted features.
- **Zhongyan Zhang:** classical models, hyperparameter tuning, and shared evaluation.
- **Zijian Zhang:** deep models, compute jobs, and interpretability analysis.
- **All members:** error analysis, research report, and presentation.

Responsibilities may change as the project develops. GitHub collaborator invitations have not been configured.

## Data sources and licensing

Data come from [Galaxy10 DECaLS](https://zenodo.org/records/10845026), compiled by Leung and Bovy; images originate from DESI Legacy Imaging Surveys and class labels from Galaxy Zoo. The dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0); see [the data record](docs/DATA.md). No software license has been selected for this repository.

The selected deduplicated file contains 17,675 images, distinct from the original release's 17,736; cleaning retains 17,655 for comparison.

The research plan is based on the team's revised proposal dated October 7, 2026. Original source documents are kept locally and are not uploaded. See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration and provenance records; the team must understand and independently verify preparation decisions and interpretations.
