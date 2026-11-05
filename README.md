# Galaxy Morphology Classification

[中文](README.zh-CN.md)

Compare hand-crafted image features, a convolutional neural network (CNN) trained from scratch, and pretrained CNNs and vision transformers on the ten-class Galaxy10 DECaLS dataset.

**Status: data preparation v1 and training-only exploratory analysis completed on November 3, 2026.** The verified raw release has 17,675 images; cleaning retains 17,655. Frozen split records are under `metadata/data_v1/`. [Exploratory findings and seven figures](docs/eda/README.md) use training images only; [the executed notebook](notebooks/01_exploration.ipynb) reproduces them. Handcrafted image features and training-only preprocessing are now complete; see [the feature record](docs/features/README.md). Classifier training and performance evaluation remain pending.

## Research questions

- How does performance change from classical models on hand-crafted features to a scratch CNN and fine-tuned pretrained models?
- Do transformers achieve higher recall on Disturbed and Merging galaxies than comparably sized CNNs?
- Which classes and images confuse different model families?
- How much does class-weighted training help minority classes?

## Planned model comparison

| Family | Models |
| --- | --- |
| Hand-crafted features | Multinomial logistic regression; k-nearest neighbors; radial basis function support vector machine; random forest; XGBoost |
| CNN trained from scratch | Custom CNN with five convolutional blocks |
| Pretrained CNNs | ResNet-50; ConvNeXt-Tiny |
| Pretrained transformers | ViT-Small/16; Swin-Tiny |

A majority-class baseline is an additional sanity check and is excluded from the ten-model count.

## Shared evaluation plan

Use the frozen original-row indices under `metadata/data_v1/` for every model. The saved split contains 12,359 training, 2,650 validation, and 2,646 test images. Integer per-class allocation gives approximately 70/15/15. Keep the test set closed during model selection. Macro F1 is the planned primary metric; three training seeds per deep model remain to be chosen. Duplicate screening has documented limits; see [the data record](docs/DATA.md).

See [the experiment plan](docs/EXPERIMENT_PLAN.md) for evaluation safeguards and target criteria.

## Repository guide

| Path | Purpose |
| --- | --- |
| [docs/DATA.md](docs/DATA.md) | Dataset source, fields, class labels, and planned access checks |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Planned methods, metrics, targets, and open decisions |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned stages, deliverables, and ownership |
| [docs/ENVIRONMENT_PLAN.md](docs/ENVIRONMENT_PLAN.md) | Proposed tools and compute environment |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaboration and experiment-recording conventions |
| [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE) | Task and experiment templates |
| [.gitignore](.gitignore) | Rules for local environments, secrets, datasets, and model artifacts |

## Local data preparation

Use Python 3.12 and the data-only dependencies in `requirements-data.txt`:

```powershell
python -m pip install -r requirements-data.txt
python check_data.py
python validate_data_preparation.py
```

`check_data.py` audits the unchanged raw file and will still flag its recorded issues. `validate_data_preparation.py` verifies the cleaned sample manifest, source checksum, saved partitions, and known-candidate separation.

- `prepare_data.py`: reproducible image screening and reviewed split construction.
- `metadata/data_v1/sample_manifest.csv`: all original row indices, labels, exclusions, metadata flags, and partition membership.
- `metadata/data_v1/{train,validation,test}_indices.json`: the indices all team members must load.
- `metadata/data_v1/review_decisions.json`: exclusions and visual review provenance.
- `metadata/data_v1/summary.json`: counts, seed, file hashes, and screening limitations.
- `data/processed/preparation_v1/`: ignored thumbnail cache and visual review sheets; source data remain in `data/raw/`.

The v1 split is frozen: `prepare_data.py` refuses to overwrite a completed version. Recreate screening in a different output/cache directory using `--scan-only`, record review decisions there, then use `--finalize`. Model training commands will be added when those modules exist.

## Exploratory analysis

```powershell
python -m pip install -r requirements-eda.txt
python eda.py
python validate_eda.py
python -m unittest discover -s tests -v
```

Open `notebooks/01_exploration.ipynb` for the executed walkthrough. The [exploratory record](docs/eda/README.md) includes class balance, four training examples per class, class-average images, stored-pixel distributions, and redshift/pixel-scale context. Small aggregate tables and curated PNG/SVG figures are tracked under `docs/eda/`; per-image diagnostics and caches remain ignored under `data/processed/eda_v1/`.

All image-derived exploration uses the 12,359 training rows. Full retained class counts come from the frozen manifest. The held-out images do not enter the exploratory statistics or gallery. No model performance or CNN-versus-transformer conclusion is reported. Feature-representation UMAP comparisons will be added with the relevant feature/model modules.

## Handcrafted features

The fixed v1 extractor covers stored-channel statistics, display-image morphology proxies, shape moments and HOG. All 17,655 retained images have 54 scalar/quality descriptors plus 1,764 raw HOG values. Training-only scalar imputation/scaling and 64-component HOG PCA/scaling produce 118 numeric inputs per image.

```powershell
python -m pip install -r requirements-features.txt
python build_features.py
python validate_features.py
python -m unittest discover -s tests -v
```

The prepared local environment is also available as `.venv\Scripts\python.exe`. See [the feature record](docs/features/README.md), [fixed configuration](configs/features_v1.json), and [executed feature audit](notebooks/02_handcrafted_features.ipynb).

Large matrices, the scalar CSV, and the fitted transformer remain in ignored `data/processed/features_v1/`; small schemas and provenance are under `metadata/features_v1/`. The scalar CSV's `original_index`, `label`, and `split` columns are metadata, **not inputs**. The NPZ `X` arrays keep them separate.

Training diagnostics found 0 undefined scalar entries and 2,082 masks touching the fixed detection boundary (16.85%). The fixed 64 HOG components retain 42.03% of standardized training texture variance. These are baseline engineering choices with documented limits, not evidence of an optimal representation or calibrated morphology.

For future cross-validation, pass **raw descriptors** to `make_model_pipeline(estimator)` from `galaxy_features.py` and put the whole pipeline inside GridSearchCV. Do not tune on the already transformed full-training exports: imputation, scaling and PCA must be refitted within each fold. The saved full-training transformer is for the final training fit and held-out inference. Training UMAP is unsupervised and descriptive; no classifier score is reported.

## Team

- **Yaru Gao:** data acquisition, cleaning, duplicate checks, shared split, exploratory analysis, and hand-crafted features.
- **Zhongyan Zhang:** classical models, tuning, and shared evaluation.
- **Zijian Zhang:** deep models, compute jobs, and interpretability.
- **All members:** error analysis, research report, and presentation.

Responsibilities may change as the project develops. GitHub collaborator invitations have not been configured.

## Data and licensing

Use [Galaxy10 DECaLS on Zenodo](https://zenodo.org/records/10845026), specifically `Galaxy10_DECals_NoDuplicated.h5`. The original release counts must not be used as the deduplicated sample counts.

Dataset attribution and its separate license are described in [docs/DATA.md](docs/DATA.md). No software license has been selected for this repository.

## Planning source

The plan is based on the team's revised research proposal dated October 7, 2026. Original source documents are retained locally and are not uploaded here. Repository setup, data preparation, visual screening, exploratory analysis, and the handcrafted-feature scripts, preprocessing, figures and notebooks were assisted by Codex. The records identify this provenance; independent team confirmation has not been recorded. The team must understand and independently verify the decisions and interpretations.
