"""Train a Galaxy10 image classifier on the frozen training/validation split."""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from deep_data import load_training_data
from deep_models import MODEL_NAMES, build_model

ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 10
CHANNELS_LAST_MODELS = {"scratch_cnn", "resnet50", "convnext_tiny"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="scratch_cnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/raw/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "metadata/data_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/deep_v1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--patience", type=int, default=5, help="0 disables early stopping")
    parser.add_argument("--lr", type=float, help="Defaults to 0.001 for scratch and 0.0003 for fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    args = parser.parse_args()
    if not args.prepare_data_only and args.output_dir is None:
        parser.error("--output-dir is required for training")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if min(args.epochs, args.batch_size, args.image_size) < 1 or min(args.workers, args.patience) < 0:
        parser.error("epochs, batch size and image size must be positive; workers and patience may be zero")
    for limit in (args.max_train_batches, args.max_val_batches):
        if limit is not None and limit < 1:
            parser.error("batch limits must be positive")
    if args.lr is None:
        args.lr = 1e-3 if args.model == "scratch_cnn" else 3e-4
    if not np.isfinite(args.lr) or not np.isfinite(args.weight_decay) or args.lr <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be finite and positive, and weight decay finite and nonnegative")
    return args


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type != "cuda":
        raise ValueError("Use a CPU or CUDA device.")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError("CUDA is unavailable on the requested device. Check the PyTorch build and GPU driver.")
    return torch.device("cuda", index)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(args, data, device):
    order_rng = torch.Generator()
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train = DataLoader(
        data.train,
        sampler=RandomSampler(data.train, generator=order_rng),
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    validation = DataLoader(
        data.validation,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_args,
    )
    return train, validation, order_rng


@contextmanager
def run_lock(output):
    """Release the lock automatically if the training process exits."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Cannot acquire exclusive run access to {output}: {error}") from error
        yield


@contextmanager
def atomic_output(path):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_checkpoint(path, payload):
    with atomic_output(path) as temporary:
        with temporary.open("wb") as file:
            torch.save(payload, file)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_predictions(path, result):
    probabilities, labels, rows = result
    predicted_labels = probabilities.argmax(axis=1)
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["original_row", "true_label", "predicted_label", *[f"p_{i}" for i in range(NUM_CLASSES)]])
            for row, label, predicted, scores in zip(rows, labels, predicted_labels, probabilities, strict=True):
                writer.writerow([int(row), int(label), int(predicted), *[format(value, ".9g") for value in scores]])


def augment_images(images):
    # The three independent operations sample uniformly from the square's eight symmetries.
    shape = (images.shape[0], 1, 1, 1)
    choices = torch.rand((3, *shape), device=images.device) < 0.5
    images = torch.where(choices[0], images.flip((-1,)), images)
    images = torch.where(choices[1], images.flip((-2,)), images)
    return torch.where(choices[2], images.transpose(-2, -1), images)


def normalize_batch(images, mean, std, device, memory_format, training=False):
    images = images.to(device, non_blocking=True)
    if training:
        images = augment_images(images)
    images = images.to(dtype=torch.float32, memory_format=memory_format)
    return images.div_(255).sub_(mean).div_(std)


def train_epoch(model, loader, optimization, normalize, device, limit, epoch):
    optimizer, loss_fn, scaler, autocast = optimization
    model.train()
    label_weight = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)
    example_count = 0
    started = perf_counter()
    steps = min(len(loader), limit) if limit is not None else len(loader)
    for step, (images, labels, _) in enumerate(loader, 1):
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(normalize(images, training=True))
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            batch_weight = loss_fn.weight[labels].sum()
            label_weight += batch_weight
            loss_sum += loss.detach() * batch_weight
        example_count += len(labels)
        if step == 1 or step % 50 == 0:
            elapsed = perf_counter() - started
            print(
                f"Epoch {epoch} training {step}/{steps}: loss {loss.item():.4f}, "
                f"{example_count / elapsed:.1f} images/s",
                flush=True,
            )
        if step == steps:
            break
    return (loss_sum / label_weight).item()


@torch.inference_mode()
def validation_batches(model, loader, normalize, autocast, limit):
    model.eval()
    rows = []
    labels = []
    logits = []
    steps = min(len(loader), limit) if limit is not None else len(loader)
    for step, (batch, targets, original_rows) in enumerate(loader, 1):
        with autocast():
            predictions = model(normalize(batch))
        logits.append(predictions.float())
        labels.append(targets)
        rows.append(original_rows)
        if step == steps:
            break
    return torch.cat(logits), torch.cat(labels), torch.cat(rows).numpy()


@torch.inference_mode()
def validate(model, loader, normalize, device, autocast, limit):
    logits, targets, rows = validation_batches(model, loader, normalize, autocast, limit)
    precision_retry = not torch.isfinite(logits).all().item()
    if precision_retry and device.type == "cuda":
        print("Non-finite validation logits; repeating validation in float32.", flush=True)
        logits, targets, rows = validation_batches(model, loader, normalize, nullcontext, limit)
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Validation produced non-finite logits at full precision.")
    loss = nn.functional.cross_entropy(logits, targets.to(device)).item()
    probabilities = logits.softmax(dim=1).cpu().numpy()
    targets = targets.numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Validation produced non-finite probabilities.")
    predictions = probabilities.argmax(axis=1)
    confusion = np.bincount(NUM_CLASSES * targets + predictions, minlength=NUM_CLASSES**2)
    confusion = confusion.reshape(NUM_CLASSES, NUM_CLASSES)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(NUM_CLASSES), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros(NUM_CLASSES), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(NUM_CLASSES), where=denominator > 0)
    metrics = {
        "loss": loss,
        "validation_float32_retry": precision_retry,
        "accuracy": float(true_positive.sum() / len(targets)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "examples": len(targets),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {"id": index, "support": int(support[index]), "precision": precision[index],
             "recall": recall[index], "f1": f1[index]}
            for index in range(NUM_CLASSES)
        ],
    }
    return metrics, (probabilities, targets, rows)


def code_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def make_settings(args, data, device):
    setting_names = (
        "model", "seed", "epochs", "batch_size", "patience", "lr", "weight_decay", "image_size", "workers",
        "no_pretrained", "max_train_batches", "max_val_batches",
    )
    return {
        **{name: getattr(args, name) for name in setting_names},
        "device": str(device),
        "split_id": data.split_id,
        "code_sha256": {
            filename: sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in ("train_deep.py", "deep_data.py", "deep_models.py")
        },
    }


def run_training(args, data, device):
    output = args.output_dir.resolve()
    config_path = output / "config.json"
    history_path = output / "history.json"
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    completion_path = output / "validation_metrics.json"
    settings = make_settings(args, data, device)
    with run_lock(output):
        saved_config = None
        if config_path.exists() and args.resume:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            inconsistent = [key for key, value in settings.items() if saved_config.get(key) != value]
            if inconsistent:
                raise ValueError(f"Current settings differ from the saved run: {', '.join(inconsistent)}.")
            if completion_path.exists():
                print(f"Run already completed: {output}", flush=True)
                return
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError(f"Output directory {output} is not empty. Use --resume for an existing run.")

        checkpoint = read_checkpoint(last_path) if last_path.exists() else None
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_amp = device.type == "cuda"
        if use_amp:
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        pretrained = args.model != "scratch_cnn" and not args.no_pretrained
        print(f"Initializing {args.model}, seed {args.seed}, on {device}", flush=True)
        bundle = build_model(
            args.model,
            pretrained=pretrained and checkpoint is None,
            scratch_mean=data.train_mean,
            scratch_std=data.train_std,
        )
        if saved_config is not None:
            config = saved_config
        else:
            config = {
                **settings,
                "code_revision": code_revision(),
                "pretrained": pretrained,
                "weights_id": bundle.weights_id,
                "weights_interpolation": bundle.interpolation,
                "mean": list(bundle.mean),
                "std": list(bundle.std),
                "smoke_test": args.no_pretrained or args.max_train_batches is not None or args.max_val_batches is not None,
                "class_counts": [int(count) for count in data.train_counts],
                "paths": {
                    "dataset": str(args.data_path.resolve()),
                    "split": str(args.split_dir.resolve()),
                    "cache": str(args.cache_dir.resolve()),
                },
                "environment": {
                    "python": sys.version.split()[0],
                    "torch": str(torch.__version__),
                    "torchvision": version("torchvision"),
                    "timm": version("timm"),
                    "h5py": version("h5py"),
                    "numpy": np.__version__,
                    "pillow": version("Pillow"),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if use_amp else None,
                },
            }
            write_json(config_path, config)

        memory_format = (
            torch.channels_last if use_amp and args.model in CHANNELS_LAST_MODELS else torch.contiguous_format
        )
        model = bundle.model.to(device, memory_format=memory_format)
        mean = torch.as_tensor(config["mean"], device=device).reshape(1, 3, 1, 1)
        std = torch.as_tensor(config["std"], device=device).reshape(1, 3, 1, 1)

        def normalize(images, training=False):
            return normalize_batch(images, mean, std, device, memory_format, training)

        def autocast():
            return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)

        counts = torch.as_tensor(data.train_counts, dtype=torch.float32, device=device)
        if len(counts) != NUM_CLASSES or (counts <= 0).any():
            raise ValueError("Training split must include all ten classes to compute class weights.")
        class_weights = counts.sum() / (NUM_CLASSES * counts)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        optimization = (optimizer, loss_fn, scaler, autocast)
        train, validation, order_rng = make_loaders(args, data, device)
        history, best_epoch, best_score = [], 0, -1.0
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if use_amp:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
            history = checkpoint["history"]
            best_epoch, best_score = checkpoint["best_epoch"], checkpoint["best_score"]
            if best_epoch == len(history):
                # Finish publishing the best checkpoint if interrupted just after saving last.pt.
                best_checkpoint = read_checkpoint(best_path) if best_path.exists() else None
                if best_checkpoint is None or best_checkpoint["epoch"] != best_epoch:
                    write_checkpoint(best_path, {"model": model.state_dict(), "epoch": best_epoch, "config": config})
            del checkpoint
            write_json(history_path, history)
            print(f"Resuming after epoch {len(history)} of {args.epochs}", flush=True)

        run_started = perf_counter()
        for epoch in range(len(history) + 1, args.epochs + 1):
            if args.patience and len(history) - best_epoch >= args.patience:
                print(f"Early stopping after epoch {len(history)}; best was epoch {best_epoch}.", flush=True)
                break
            order_rng.manual_seed(args.seed + epoch)
            started = perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch(model, train, optimization, normalize, device, args.max_train_batches, epoch)
            metrics, _ = validate(model, validation, normalize, device, autocast, args.max_val_batches)
            scheduler.step()
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": metrics["loss"],
                "validation_float32_retry": metrics["validation_float32_retry"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "learning_rate": lr,
                "seconds": perf_counter() - started,
            })
            improved = metrics["macro_f1"] > best_score
            if improved:
                best_score, best_epoch = metrics["macro_f1"], epoch
            state = model.state_dict()
            write_checkpoint(last_path, {
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if use_amp else None,
                "config": config,
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
            })
            if improved:
                write_checkpoint(best_path, {"model": state, "epoch": best_epoch, "config": config})
            write_json(history_path, history)
            print(
                f"Epoch {epoch}/{args.epochs}: train loss {train_loss:.4f} | "
                f"validation macro-F1 {metrics['macro_f1']:.4f} | {history[-1]['seconds']:.1f}s",
                flush=True,
            )

        best_checkpoint = read_checkpoint(best_path)
        model.load_state_dict(best_checkpoint["model"])
        del best_checkpoint
        results, predictions = validate(model, validation, normalize, device, autocast, args.max_val_batches)
        results.update({
            "model": args.model,
            "seed": args.seed,
            "split_id": data.split_id,
            "epoch": best_epoch,
            "epochs_completed": len(history),
            "epoch_compute_seconds": sum(item["seconds"] for item in history),
            "session_seconds": perf_counter() - run_started,
            "smoke_test": config["smoke_test"],
        })
        write_predictions(output / "validation_predictions.csv", predictions)
        write_json(completion_path, results)
        print(f"Best validation macro-F1 {results['macro_f1']:.4f} at epoch {best_epoch}: {output}", flush=True)


def main():
    args = parse_args()
    print(f"Preparing frozen training and validation images at {args.image_size} pixels", flush=True)
    data = load_training_data(args.data_path, args.split_dir, args.cache_dir, args.image_size)
    print(
        f"Shared split {data.split_id}: {len(data.train)} train and {len(data.validation)} validation images.",
        flush=True,
    )
    if not args.prepare_data_only:
        run_training(args, data, resolve_device(args.device))


if __name__ == "__main__":
    main()
