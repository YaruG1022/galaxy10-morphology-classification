# Galaxy Morphology Classification

Compare hand-crafted image features, a convolutional neural network (CNN) trained from scratch, and pretrained CNNs and vision transformers on the ten-class Galaxy10 DECaLS dataset.

**Status: planning and repository setup only.** No data has been downloaded, no analytical code has been implemented, and no experiments or results are available.

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

Use the deduplicated dataset, check for remaining duplicates, then save one stratified 70/15/15 training/validation/test split. All models use the same split. Macro F1 is the primary metric; deep models have three planned training seeds. These are planned procedures, not completed work.

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

Implementation directories will be added when implementation begins. There are no installation or training commands yet.

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

The plan is based on the team's revised research proposal dated October 7, 2026. Original source documents are retained locally and are not uploaded here. Repository setup and document organization were assisted by Codex; the team will carry out and verify the analytical work.
