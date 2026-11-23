# Environment plan

**Status: local data preparation, EDA and handcrafted-feature dependencies verified on Python 3.12.4.** See `requirements-data.txt`, `requirements-eda.txt` and `requirements-features.txt`. The exploration and feature-audit notebooks have been executed locally. A project `.venv` supplies the pinned feature dependencies without changing the base installation; it was created with system-site-packages access to reuse already pinned base packages. The deep-model trainer was validated locally on a GeForce RTX 4060 Laptop GPU with PyTorch 2.8.0, torchvision 0.23.0 and CUDA 12.8. See [deep-model training](DEEP_TRAINING.md). Kamiak access and data transfer have not been validated.

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

The local data/EDA/feature requirements pin the versions actually used. Feature v1 uses scikit-image 0.25.2, scikit-learn 1.6.1 and umap-learn 0.5.9.post2; detailed runtime versions are recorded in `metadata/features_v1/summary.json`. For the Windows GPU environment, follow the pinned training setup in [deep-model training](DEEP_TRAINING.md).

## Planned implementation layout

The classifier package layout remains a proposal. Data, feature and deep-model scripts live at the repository root. Tests, notebooks, frozen sample indices and the fixed feature configuration remain in their existing directories. The following layout was proposed for expansion:

- `src/`: data preparation, features, models, and evaluation.
- `notebooks/`: exploration and analysis notebooks.
- `configs/`: reviewed model and run configurations.
- `scripts/`: entry points and compute-job submission scripts.
- `tests/`: checks for data partitioning, label alignment, and evaluation.
- `metadata/`: small, reviewed split and reproducibility records.
- `data/`, `artifacts/`, `checkpoints/`, `runs/`, `results/`: local or external datasets and generated artifacts; ignored by Git.

Small curated figures and records are tracked under `docs/eda/` and `docs/features/`, with executed walkthroughs in `notebooks/`. Large per-image diagnostics, feature matrices, fitted preprocessing objects and raw data remain under ignored `data/`. Deep-model run artifacts are written under the ignored `runs/` directory.
