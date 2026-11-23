"""Run the fixed three-seed sweep for Galaxy10 deep-model modules 5–7."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEARNING_RATES = {
    "scratch_cnn": 0.001,
    "resnet50": 0.0002,
    "convnext_tiny": 0.0002,
    "vit_small": 0.0002,
    "swin_tiny": 0.0002,
}
SEEDS = (42, 123, 2026)
RUN_PRODUCTS = (
    "best.pt",
    "last.pt",
    "history.json",
    "validation_metrics.json",
    "validation_predictions.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "deep_v1")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/raw/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "metadata/data_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/deep_v1")
    parser.add_argument("--models", nargs="+", choices=tuple(LEARNING_RATES), default=list(LEARNING_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(ROOT / "tmp" / "deep-huggingface"))
    environment.setdefault("TORCH_HOME", str(ROOT / "tmp" / "deep-torch"))
    environment.setdefault("HF_HUB_DISABLE_XET", "1")

    for model in dict.fromkeys(args.models):
        for seed in dict.fromkeys(args.seeds):
            run_dir = output_dir / model / f"seed_{seed}"
            config_path = run_dir / "config.json"
            expected = {
                "model": model,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "lr": LEARNING_RATES[model],
                "image_size": 224,
                "no_pretrained": False,
                "smoke_test": False,
            }
            if config_path.is_file():
                recorded = json.loads(config_path.read_text(encoding="utf-8"))
                differences = [key for key, value in expected.items() if recorded.get(key) != value]
                if differences:
                    raise SystemExit(f"{run_dir}: use a different output directory for changed settings: {', '.join(differences)}")
                if all((run_dir / artifact).is_file() for artifact in RUN_PRODUCTS):
                    print(f"Already complete: {model}, seed {seed}", flush=True)
                    continue

            command = [
                sys.executable, "-u", str(ROOT / "train_deep.py"),
                "--model", model,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--lr", str(LEARNING_RATES[model]),
                "--image-size", "224",
                "--workers", str(args.workers),
                "--device", args.device,
                "--data-path", str(args.data_path.resolve()),
                "--split-dir", str(args.split_dir.resolve()),
                "--cache-dir", str(args.cache_dir.resolve()),
                "--output-dir", str(run_dir),
            ]
            if config_path.is_file() or (run_dir / "last.pt").is_file():
                command.append("--resume")
            print(f"Running {model}, seed {seed}", flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
