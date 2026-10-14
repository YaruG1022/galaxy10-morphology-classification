# Environment plan

**Status: proposed tools only.** No environment has been installed, configured, or validated.

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

The team will select compatible Python, CUDA, PyTorch, and package versions after checking the available compute environment. Capture exact dependencies and relevant device information once verified. No dependency file is supplied yet because the environment is undecided.

## Planned implementation layout

These directories are a proposal, not an implemented package:

- `src/`: data preparation, features, models, and evaluation.
- `notebooks/`: exploration and analysis notebooks.
- `configs/`: reviewed model and run configurations.
- `scripts/`: entry points and compute-job submission scripts.
- `tests/`: checks for data partitioning, label alignment, and evaluation.
- `metadata/`: small, reviewed split and reproducibility records.
- `data/`, `artifacts/`, `checkpoints/`, `runs/`, `results/`: local or external datasets and generated artifacts; ignored by Git.

Keep small curated figures and result tables under a deliberately tracked location such as `docs/results/` when evidence exists. No directories containing implementation placeholders are created at this stage.
