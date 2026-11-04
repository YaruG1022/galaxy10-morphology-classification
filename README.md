# Galaxy Morphology Classification

Compare hand-crafted image features, a convolutional neural network (CNN) trained from scratch, and pretrained CNNs and vision transformers on the ten-class Galaxy10 DECaLS dataset.

**Status: local data preparation v1 completed on November 3, 2026.** The verified raw release has 17,675 images; the saved cleaning decisions retain 17,655. The fixed 70/15/15 split (seed 42) and independent integrity checks are available under `metadata/data_v1/`. No feature table, trained model, model evaluation, or research-performance result is available.

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

The plan is based on the team's revised research proposal dated October 7, 2026. Original source documents are retained locally and are not uploaded here. Repository setup, data-checking/preparation scripts, and the recorded visual screening were assisted by Codex. The review files identify this provenance; independent team confirmation has not been recorded. The team must understand and independently verify the data decisions and subsequent analytical work.
