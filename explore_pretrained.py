"""Project frozen pretrained Galaxy10 training embeddings with unsupervised UMAP."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from umap import UMAP

from deep_data import load_training_data
from deep_models import MODEL_NAMES, build_model
from eda import CLASSES, save_csv, save_json
from train_deep import ROOT, code_revision, normalize_batch, resolve_device


PRETRAINED_MODELS = MODEL_NAMES[1:]
UMAP_SETTINGS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
    "n_epochs": 200,
    "n_jobs": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=PRETRAINED_MODELS, default=("vit_small", "swin_tiny"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/raw/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "metadata/data_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/deep_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/deep_representation_v1")
    args = parser.parse_args()
    if args.samples < 20 or args.batch_size < 1:
        parser.error("--samples must be at least 20 and --batch-size must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if len(args.models) != len(set(args.models)):
        parser.error("--models must list each architecture at most once")
    return args


@torch.inference_mode()
def extract_embeddings(bundle, batches, device):
    model = bundle.model.to(device).eval()
    mean = torch.tensor(bundle.mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(bundle.std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    embeddings = []
    for images, _, _ in batches:
        normalized = normalize_batch(images, mean, std, device, torch.contiguous_format)
        features = model.forward_features(normalized)
        vectors = model.forward_head(features, pre_logits=True)
        if vectors.ndim != 2 or vectors.shape[0] != images.shape[0]:
            raise ValueError("Expected one pooled embedding vector per image")
        embeddings.append(vectors.to(device="cpu", dtype=torch.float32).numpy())
    features = np.concatenate(embeddings)
    if not np.isfinite(features).all():
        raise ValueError("Pretrained embeddings contain non-finite values")
    return features


def plot_projection(path, coordinates, labels, model, seed):
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(figsize=(9.5, 6.3), layout="constrained")
    try:
        for class_id, name in enumerate(CLASSES):
            points = coordinates[labels == class_id]
            axes.scatter(points[:, 0], points[:, 1], s=9, alpha=0.6,
                         color=colors[class_id], label=f"{class_id}: {name}", linewidths=0)
        axes.set(
            title=f"{model} | Frozen pretrained features",
            xlabel="UMAP 1 (arbitrary units)",
            ylabel="UMAP 2 (arbitrary units)",
        )
        axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=1.7, frameon=False)
        figure.supxlabel(
            f"{len(labels):,} training images · UMAP seed {seed} · Class labels used for color only",
            fontsize=9,
        )
        figure.savefig(path, dpi=170)
    finally:
        plt.close(figure)


def save_projection(destination, coordinates, labels, rows, metadata):
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "projection"
        staged.mkdir()
        save_csv(staged / "projection.csv", [
            {"original_row": int(row), "class_id": int(label), "umap_1": float(x), "umap_2": float(y)}
            for row, label, (x, y) in zip(rows, labels, coordinates, strict=True)
        ])
        plot_projection(staged / "projection.png", coordinates, labels, metadata["model"], metadata["sampling_seed"])
        save_json(staged / "metadata.json", metadata)
        staged.rename(destination)


def main():
    plt.switch_backend("Agg")
    args = parse_args()
    for name in args.models:
        destination = args.output_dir / name
        if destination.exists():
            raise FileExistsError(f"{destination} already exists. Choose another --output-dir for a new projection.")

    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    data = load_training_data(args.data_path, args.split_dir, args.cache_dir, image_size=224)
    if args.samples > len(data.train):
        raise ValueError(f"Requested {args.samples} images from a training set of {len(data.train)}")
    indices = np.random.default_rng(args.seed).choice(len(data.train), size=args.samples, replace=False)
    indices.sort()
    batches = DataLoader(
        Subset(data.train, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    settings = {**UMAP_SETTINGS, "random_state": args.seed}
    versions = {
        package: version(package) for package in (
            "torch", "torchvision", "timm", "Pillow", "numpy", "umap-learn", "numba", "llvmlite",
            "pynndescent", "scikit-learn", "scipy", "matplotlib",
        )
    }
    revision = code_revision()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        print(f"{name}: extracting frozen embeddings for {args.samples:,} training images", flush=True)
        bundle = build_model(name, pretrained=True, scratch_mean=data.train_mean, scratch_std=data.train_std)
        weights = bundle.weights_id
        normalization = {"mean": bundle.mean, "std": bundle.std}
        features = extract_embeddings(bundle, batches, device)
        del bundle
        print(f"{name}: fitting two-dimensional UMAP", flush=True)
        coordinates = UMAP(**settings).fit_transform(features)
        if coordinates.shape != (args.samples, 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid UMAP coordinates for {name}")

        # Class labels are consulted only after the unsupervised projection.
        labels = data.train.labels[indices]
        rows = data.train.rows[indices]
        metadata = {
            "model": name,
            "pretrained_weights": weights,
            "embedding_layer": "forward_features followed by forward_head(pre_logits=True)",
            "embedding_shape": list(features.shape),
            "data_split_id": data.split_id,
            "partition": "train",
            "training_images_available": len(data.train),
            "sample_count": len(indices),
            "sampling_method": "uniform sampling without replacement over frozen train positions",
            "sampling_seed": args.seed,
            "preprocessing": {
                "image_size": [224, 224],
                "resize": "square resize with Pillow bilinear interpolation",
                "normalization": normalization,
            },
            "umap": settings,
            "batch_size": args.batch_size,
            "device": str(device),
            "software": versions,
            "code_revision": revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        destination = args.output_dir / name
        save_projection(destination, coordinates, labels, rows, metadata)
        print(f"{name}: wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
