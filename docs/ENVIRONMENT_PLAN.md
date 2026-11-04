# Environment plan

**Status: local data preparation, EDA and handcrafted-feature dependencies verified on Python 3.12.4.** See `requirements-data.txt`, `requirements-eda.txt` and `requirements-features.txt`. The exploration and feature-audit notebooks have been executed locally. A project `.venv` supplies the pinned feature dependencies without changing the base installation; it was created with system-site-packages access to reuse already pinned base packages. Kamiak access, GPU training, CUDA compatibility and the future modeling stack have not been validated.

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

The local data/EDA/feature requirements pin the versions actually used. Feature v1 uses scikit-image 0.25.2, scikit-learn 1.6.1 and umap-learn 0.5.9.post2; detailed runtime versions are recorded in `metadata/features_v1/summary.json`. The team still needs to select compatible CUDA, PyTorch and model-training package versions after checking the intended compute environment. Local data exploration does not verify a GPU training environment.

## Planned implementation layout

The classifier package layout remains a proposal. Root-level data/EDA/feature scripts, `tests/`, `notebooks/`, `metadata/` and the fixed feature configuration in `configs/` now exist; `src/` and classifier/deep-model configurations are not implemented:

- `src/`: data preparation, features, models, and evaluation.
- `notebooks/`: exploration and analysis notebooks.
- `configs/`: reviewed model and run configurations.
- `scripts/`: entry points and compute-job submission scripts.
- `tests/`: checks for data partitioning, label alignment, and evaluation.
- `metadata/`: small, reviewed split and reproducibility records.
- `data/`, `artifacts/`, `checkpoints/`, `runs/`, `results/`: local or external datasets and generated artifacts; ignored by Git.

Small curated figures and English records are tracked under `docs/eda/` and `docs/features/`, with executed walkthroughs in `notebooks/`. Large per-image diagnostics, feature matrices, fitted preprocessing objects and raw data remain under ignored `data/`. Classifier directories will be added when that module begins.
