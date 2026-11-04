# Environment plan

**Status: local data preparation and EDA dependencies verified on Python 3.12.4.** See `requirements-data.txt` and `requirements-eda.txt`. The exploration notebook has been executed locally. Kamiak access, GPU training, CUDA compatibility and the future modeling stack have not been validated.

| Purpose | Proposed tools |
| --- | --- |
| Language and exploration | Python 3; Jupyter notebooks |
| Dataset access | h5py |
| Image processing and duplicate checks | scikit-image; ImageHash |
| Classical models and evaluation | scikit-learn; XGBoost |
| Deep learning | PyTorch; torchvision; timm |
| Interpretability | pytorch-grad-cam; attention rollout |
| Visualization | Matplotlib; seaborn; umap-learn |
| Research compute | Kamiak GPU resources and Slurm jobs |

The local data/EDA requirements pin the versions actually used. The team still needs to select compatible CUDA, PyTorch and model-training package versions after checking the intended compute environment. Local data exploration does not verify a GPU training environment.

## Planned implementation layout

The modeling package layout remains a proposal. Root-level data/EDA scripts, `tests/`, `notebooks/`, and `metadata/` now exist; `src/` and model configurations are not implemented:

- `src/`: data preparation, features, models, and evaluation.
- `notebooks/`: exploration and analysis notebooks.
- `configs/`: reviewed model and run configurations.
- `scripts/`: entry points and compute-job submission scripts.
- `tests/`: checks for data partitioning, label alignment, and evaluation.
- `metadata/`: small, reviewed split and reproducibility records.
- `data/`, `artifacts/`, `checkpoints/`, `runs/`, `results/`: local or external datasets and generated artifacts; ignored by Git.

Small curated exploratory figures and aggregate tables are tracked under `docs/eda/`, with an executed walkthrough in `notebooks/01_exploration.ipynb`. Large per-image diagnostics and all raw data remain under ignored `data/`. Modeling directories will be added only when implementation begins.
