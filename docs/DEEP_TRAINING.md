# Galaxy10 deep-model training

Modules 5–7 use the same frozen `metadata/data_v1/` rows from the original
`data/raw/Galaxy10_DECals_NoDuplicated.h5` file.

| Module | Runner names | Initialization |
| --- | --- | --- |
| 5 | `scratch_cnn` | Five custom convolutional blocks |
| 6 | `resnet50`, `convnext_tiny` | `resnet50.a1_in1k`, `convnext_tiny.fb_in1k` |
| 7 | `vit_small`, `swin_tiny` | `vit_small_patch16_224.augreg_in1k`, `swin_tiny_patch4_window7_224.ms_in1k` |

## Environment

Use Python 3.12 in a separate virtual environment. Install matching PyTorch
and torchvision builds for your operating system and driver. The Windows
GPU run uses [PyTorch 2.8.0 and torchvision 0.23.0 with CUDA 12.8](https://pytorch.org/get-started/previous-versions/):

```text
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-deep.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The first pretrained run needs access to the selected public weight files.
The suite uses local caches under `tmp/` unless their environment variables
are set. It needs approximately 2.2 GiB for the 224-pixel image cache and
additional disk space for two checkpoints per model and seed.

## Run and reproduce

```text
python run_deep_suite.py
python run_deep_suite.py --models vit_small swin_tiny
python train_deep.py --help
```

The full suite uses training seeds 42, 123, and 2026. It runs up to 12 epochs
with early stopping after four validation epochs without a macro F1 improvement.
It uses a batch size of 32, a learning rate of 0.001 for the scratch model and
0.0002 for the pretrained models. For a different budget, use a separate output
directory and record the settings with the resulting scores. Rerunning the same
suite skips finished runs and resumes from the latest saved checkpoint when
available.

All models use the same resize to 224 × 224. The scratch model uses channel
statistics fitted on training images. Pretrained models use the channel
normalization of their selected weights. Rotations and flips apply to training
batches. Class weights also come from training labels only. The frozen test split
is held back for final evaluation in module 8.

A finished `runs/deep_v1/<model>/seed_<seed>/` contains training configuration
and provenance, epoch history, latest and best checkpoints, validation metrics,
and per-image validation predictions keyed by original HDF5 row. Checkpoint
selection uses validation macro F1. If mixed-precision validation produces
non-finite logits, the trainer repeats validation in float32 and records the retry.
Retain the run files when sharing numerical
results, since Git ignores the large run directory. The completed local suite also has a five-model
summary in `runs/deep_v1/validation_summary.csv`.

## Inspect pretrained representations

```text
python explore_pretrained.py
```

The default comparison projects upstream ViT-Small/16 and Swin-Tiny
embeddings before Galaxy10 fine-tuning. Both use the same 2,000 randomly
selected training rows, with seed 42. UMAP fits on embeddings alone. The
plot applies class colors afterward. The coordinates, original row IDs,
figure, and settings are written under `runs/deep_representation_v1/`.
