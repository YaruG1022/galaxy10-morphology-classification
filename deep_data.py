"""Frozen Galaxy10 train/validation images and training-only normalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


_CLASS_COUNT = 10
_PARTITIONS = ("train", "validation", "test")
_CACHE_FORMAT = "rgb-uint8-chw-pillow-bilinear-v1"
_BATCH_ROWS = 32


class GalaxyImageDataset(Dataset):
    """Map one fixed split to cached CHW uint8 images, labels and original IDs."""

    def __init__(self, image_file: Path, start: int, rows: np.ndarray, labels: np.ndarray):
        self.image_file = image_file
        self.start = start
        self.rows = rows
        self.labels = labels
        self._images = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._images is None:
            self._images = np.load(self.image_file, mmap_mode="r", allow_pickle=False)
        # A writable copy prevents individual transforms from touching the shared cache.
        image = torch.from_numpy(self._images[self.start + index].copy())
        return image, int(self.labels[index]), int(self.rows[index])

    def __getstate__(self) -> dict:
        # Workers open the mapped file locally instead of pickling its contents.
        return {**self.__dict__, "_images": None}


@dataclass(frozen=True)
class DeepData:
    train: GalaxyImageDataset
    validation: GalaxyImageDataset
    train_counts: tuple[int, ...]
    train_mean: tuple[float, ...]
    train_std: tuple[float, ...]
    split_id: str


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_splits(split_dir: Path, data_path: Path, file_size: int) -> tuple[dict, str]:
    summary_path = split_dir / "summary.json"
    summary = json.loads(summary_path.read_bytes()) if summary_path.is_file() else {}
    if summary.get("raw_size_bytes", file_size) != file_size:
        raise ValueError(f"Image file size does not match the frozen dataset: {data_path}")

    splits = {}
    manifest_hashes = summary.get("artifact_sha256", {})
    index_hashes = {}
    for name in _PARTITIONS:
        filename = f"{name}_indices.json"
        content = (split_dir / filename).read_bytes()
        index_hashes[filename] = sha256(content).hexdigest()
        if filename in manifest_hashes and index_hashes[filename] != manifest_hashes[filename]:
            raise ValueError(f"{filename} does not match the frozen split checksum")
        rows = json.loads(content)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{filename} must contain a nonempty list of original row IDs")
        if any(type(row) is not int or not 0 <= row < 2**63 for row in rows):
            raise ValueError(f"{filename} contains an invalid original row ID")
        splits[name] = np.asarray(rows, dtype=np.int64)

    all_rows = np.concatenate(list(splits.values()))
    if np.unique(all_rows).size != all_rows.size:
        raise ValueError("Frozen splits contain duplicate or overlapping original row IDs")

    source = summary.get("raw_md5") or {"filename": data_path.name, "size": file_size}
    split_id = _fingerprint({"source": source, "split_indices": index_hashes})
    return splits, split_id


def _build_cache(folder: Path, data_path: Path, splits: dict, image_size: int) -> None:
    train_size = len(splits["train"])
    rows = np.concatenate([splits["train"], splits["validation"]])
    sorted_positions = np.argsort(rows)
    sorted_rows = rows[sorted_positions]
    run_boundaries = np.flatnonzero(np.diff(sorted_rows) != 1) + 1
    pixel_sums = np.zeros(3, dtype=np.uint64)
    pixel_squared_sums = np.zeros(3, dtype=np.uint64)

    # One 553-row compressed block of the source spans about 104 MiB across
    # its spatial/color chunks. Keeping it cached avoids repeat decompression
    # while reading contiguous runs of allowed rows around the held-out IDs.
    with h5py.File(data_path, "r", rdcc_nbytes=128 * 1024**2, rdcc_nslots=10007) as source:
        images = source["images"]
        labels = source["ans"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError("Galaxy10 images must be stored as RGB uint8 arrays")
        if labels.shape != (len(images),) or labels.dtype.kind not in "ui":
            raise ValueError("Galaxy10 labels must be a vector of integers aligned with images")
        if max(int(values.max()) for values in splits.values()) >= len(images):
            raise ValueError("Frozen split row ID is outside the Galaxy10 image file")

        sorted_labels = labels[sorted_rows]
        if np.any(sorted_labels < 0) or np.any(sorted_labels >= _CLASS_COUNT):
            raise ValueError("Galaxy10 labels must be in the range 0 through 9")
        partition_labels = np.empty(len(rows), dtype=np.int64)
        partition_labels[sorted_positions] = sorted_labels

        cached_images = np.lib.format.open_memmap(
            folder / "images.npy", mode="w+", dtype=np.uint8,
            shape=(len(rows), 3, image_size, image_size),
        )
        for run_positions, run_rows in zip(np.split(sorted_positions, run_boundaries),
                                           np.split(sorted_rows, run_boundaries), strict=True):
            for batch_start in range(0, len(run_rows), _BATCH_ROWS):
                positions = run_positions[batch_start:batch_start + _BATCH_ROWS]
                first = int(run_rows[batch_start])
                originals = images[first:first + len(positions)]
                if originals.shape[1:3] == (image_size, image_size):
                    resized = originals
                else:
                    resized = np.stack([
                        np.asarray(Image.fromarray(image).resize(
                            (image_size, image_size), Image.Resampling.BILINEAR,
                        ))
                        for image in originals
                    ])
                batch = np.moveaxis(resized, -1, 1)
                cached_images[positions] = batch
                train_batch = batch[positions < train_size]
                if len(train_batch):
                    pixel_sums += train_batch.sum(axis=(0, 2, 3), dtype=np.uint64)
                    squared = np.square(train_batch, dtype=np.uint32)
                    pixel_squared_sums += squared.sum(axis=(0, 2, 3), dtype=np.uint64)
        cached_images.flush()
        del cached_images

    pixels_per_channel = train_size * image_size**2
    mean = pixel_sums / (pixels_per_channel * 255)
    second_moment = pixel_squared_sums / (pixels_per_channel * 255**2)
    std = np.sqrt(np.maximum(second_moment - mean**2, 0))
    counts = np.bincount(partition_labels[:train_size], minlength=_CLASS_COUNT)
    np.save(folder / "labels.npy", partition_labels, allow_pickle=False)
    (folder / "stats.json").write_text(json.dumps({
        "train_counts": counts.tolist(), "train_mean": mean.tolist(), "train_std": std.tolist(),
    }), encoding="utf-8")


def load_training_data(
    data_path: Path, split_dir: Path, cache_dir: Path, image_size: int = 224,
) -> DeepData:
    """Load train/validation images without reading held-out images or labels.

    Cache construction resizes directly to a square with Pillow bilinear
    interpolation and no augmentation. The three-channel statistics use all
    resized training pixels on a [0, 1] scale and population standard deviation.
    The local cache is disposable and keyed to the frozen indices, resolution,
    cache format and source file path, size and modification time.
    """
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    data_path = Path(data_path).resolve(strict=True)
    stat = data_path.stat()
    splits, split_id = _read_splits(Path(split_dir), data_path, stat.st_size)
    cache_dir = Path(cache_dir).resolve()
    cache_key = _fingerprint({
        "split_id": split_id, "source": str(data_path), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "image_size": image_size, "format": _CACHE_FORMAT,
    })
    folder = cache_dir / cache_key
    if not folder.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".building-", dir=cache_dir) as building:
            built = Path(building)
            _build_cache(built, data_path, splits, image_size)
            updated = data_path.stat()
            if (updated.st_size, updated.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("The Galaxy10 image file changed while the cache was being built")
            try:
                built.rename(folder)
            except OSError:
                if not folder.is_dir():
                    raise
                # A concurrent process already published a complete cache.

    image_file = folder / "images.npy"
    labels = np.load(folder / "labels.npy", allow_pickle=False)
    stats = json.loads((folder / "stats.json").read_bytes())
    train_size = len(splits["train"])
    return DeepData(
        train=GalaxyImageDataset(image_file, 0, splits["train"], labels[:train_size]),
        validation=GalaxyImageDataset(image_file, train_size, splits["validation"], labels[train_size:]),
        train_counts=tuple(stats["train_counts"]),
        train_mean=tuple(stats["train_mean"]),
        train_std=tuple(stats["train_std"]),
        split_id=split_id,
    )
