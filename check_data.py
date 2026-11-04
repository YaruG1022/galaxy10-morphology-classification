"""Read-only Galaxy10 checks. Authored with Codex assistance.

Run from PowerShell: python check_data.py
Requires numpy and h5py. Does not clean, split, plot, or train on data.
Exact duplicates are flagged for review, never removed automatically.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

OFFICIAL_MD5 = "a920094d5e470e3705f4691c0d6dff54"
SOURCE = "https://zenodo.org/records/10845026"
FIELDS = ("ans", "images", "ra", "dec", "pxscale", "redshift")


def audit(data_path, expected_md5, batch_size):
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": str(data_path), "source": SOURCE,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "h5py": h5py.__version__},
        "checks": [], "details": {},
        "not_checked": [
            "Near-duplicate images, including rotated/rescaled views of one galaxy",
            "Nearby sky coordinates and their interpretation as duplicate objects",
            "Label correctness against images and astronomical ground truth",
            "Undocumented finite missing-value sentinels in metadata",
            "Train/validation/test leakage (no split is created or audited)",
        ],
    }

    def check(name, passed, message, warning=False):
        status = "PASS" if passed else ("WARN" if warning else "FAIL")
        report["checks"].append({"name": name, "status": status, "message": message})
        print(f"[{status}] {name}: {message}", flush=True)

    try:
        report["size_bytes"] = data_path.stat().st_size
        print("Reading file checksum...", flush=True)
        digest = hashlib.md5()
        with data_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        report["md5"] = actual
        report["expected_md5"] = expected_md5.lower()
        check("file_checksum", actual == expected_md5.lower(), actual)
        if actual != expected_md5.lower():
            return report

        with h5py.File(data_path, "r") as f:
            missing = [k for k in FIELDS if k not in f or not isinstance(f[k], h5py.Dataset)]
            check("required_datasets", not missing, f"Missing/non-dataset fields: {missing}")
            if missing:
                return report
            schema = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in FIELDS}
            report["details"]["schema"] = schema
            images = f["images"]
            n = images.shape[0] if images.ndim else 0
            report["sample_count"] = n
            structure_ok = n > 0 and images.shape == (n, 256, 256, 3)
            structure_ok = structure_ok and all(f[k].shape == (n,) for k in FIELDS if k != "images")
            check("shapes_and_lengths", structure_ok, json.dumps(schema))
            if not structure_ok:
                return report
            types_ok = images.dtype == np.dtype("uint8") and f["ans"].dtype == np.dtype("uint8")
            types_ok = types_ok and all(f[k].dtype == np.dtype("float64") for k in FIELDS if k not in ("ans", "images"))
            check("expected_dtypes", types_ok, "Expected uint8 images/labels and float64 metadata")
            if not types_ok:
                return report

            labels = f["ans"][:]
            ids, counts = np.unique(labels, return_counts=True)
            class_counts = {str(int(k)): int(v) for k, v in zip(ids, counts)}
            report["details"]["class_counts"] = class_counts
            check("labels_0_to_9", np.array_equal(ids, np.arange(10)), f"Observed labels: {ids.tolist()}")
            check("class_count_total", int(counts.sum()) == n, f"{class_counts}; total={int(counts.sum())}")
            report["details"]["largest_to_smallest_class_ratio"] = float(counts.max() / counts.min())

            metadata = {}
            for key in ("ra", "dec", "pxscale", "redshift"):
                values = f[key][:]
                finite = np.isfinite(values)
                valid = values[finite]
                stats = {
                    "nan_count": int(np.isnan(values).sum()),
                    "infinite_count": int(np.isinf(values).sum()),
                    "zero_count": int((values == 0).sum()),
                    "negative_count": int((values < 0).sum()),
                    "finite_min": float(valid.min()) if valid.size else None,
                    "finite_max": float(valid.max()) if valid.size else None,
                    "first_ten": [float(x) if np.isfinite(x) else None for x in values[:10]],
                }
                metadata[key] = stats
                check(f"{key}_finite", bool(finite.all()), json.dumps(stats), warning=True)
                if key == "ra":
                    bad = (values < 0) | (values >= 360)
                elif key == "dec":
                    bad = (values < -90) | (values > 90)
                elif key == "pxscale":
                    bad = values <= 0
                else:
                    # Negative redshift may be physical or a sentinel: flag, do not reject.
                    bad = values < 0
                bad_count = int((finite & bad).sum())
                check(f"{key}_range_review", bad_count == 0,
                      f"Flagged finite values: {bad_count}; no values removed", warning=True)
            report["details"]["metadata"] = metadata

            ra, dec = f["ra"][:], f["dec"][:]
            valid_indices = np.flatnonzero(np.isfinite(ra) & np.isfinite(dec))
            coords = np.column_stack((ra[valid_indices], dec[valid_indices]))
            _, inverse, coord_counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
            coord_groups = [valid_indices[inverse == i].tolist() for i in np.flatnonzero(coord_counts > 1)]
            report["details"]["identical_coordinate_groups"] = coord_groups
            check("identical_coordinates", not coord_groups,
                  f"Groups requiring review: {len(coord_groups)}", warning=True)

            seen, duplicates = {}, {}
            pixel_min, pixel_max = 255, 0
            all_zero, constant = [], []
            print("Scanning ALL images in small batches; checking exact duplicates...", flush=True)
            for start in range(0, n, batch_size):
                batch = images[start:start + batch_size]
                pixel_min = min(pixel_min, int(batch.min()))
                pixel_max = max(pixel_max, int(batch.max()))
                for offset, img in enumerate(batch):
                    index = start + offset
                    if not img.any():
                        all_zero.append(index)
                    if img.min() == img.max():
                        constant.append(index)
                    image_hash = hashlib.sha256(img.tobytes()).hexdigest()
                    if image_hash in seen:
                        duplicates.setdefault(image_hash, [seen[image_hash]]).append(index)
                    else:
                        seen[image_hash] = index
                if (start // batch_size) % 32 == 0 or start + len(batch) == n:
                    print(f"  Images checked: {start + len(batch)}/{n}", flush=True)
            groups = list(duplicates.values())
            conflicts = [g for g in groups if np.unique(labels[g]).size > 1]
            report["details"]["images"] = {
                "checked_images": n, "pixel_min": pixel_min, "pixel_max": pixel_max,
                "all_zero_indices": all_zero, "constant_image_indices": constant,
                "exact_duplicate_groups": groups, "conflicting_label_duplicate_groups": conflicts,
            }
            check("pixel_range", 0 <= pixel_min <= pixel_max <= 255, f"[{pixel_min}, {pixel_max}]")
            check("constant_images", not constant, f"Constant: {len(constant)}; all-zero: {len(all_zero)}", warning=True)
            check("exact_duplicate_images", not groups, f"Identical-pixel groups: {len(groups)}", warning=True)
            check("duplicate_label_conflicts", not conflicts, f"Conflicting-label groups: {len(conflicts)}", warning=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        check("audit_execution", False, f"{type(exc).__name__}: {exc}")
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/raw/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--output-dir", type=Path, default=root / "data/raw/metadata")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-md5", default=OFFICIAL_MD5,
                        help="Override only for a different independently verified file")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    report = audit(args.data.resolve(), args.expected_md5, args.batch_size)
    statuses = Counter(c["status"] for c in report["checks"])
    report["summary"] = {k: statuses[k] for k in ("PASS", "WARN", "FAIL")}
    report["overall"] = "FAIL" if statuses["FAIL"] else ("REVIEW" if statuses["WARN"] else "CHECKS_PASSED")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"data_check_{stamp}.json"
    text_path = json_path.with_suffix(".txt")
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = [f"File: {report['file']}", f"Overall: {report['overall']}",
             f"Summary: {report['summary']}", ""]
    lines.extend(f"[{c['status']}] {c['name']}: {c['message']}" for c in report["checks"])
    lines.extend(["", "NOT CHECKED:", *report["not_checked"],
                  "", "PASS only covers listed checks; warnings require human review.",
                  "No source data was changed. JSON contains detailed indices and counts."])
    with text_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"\nOverall: {report['overall']} {report['summary']}")
    print(f"Reports:\n{text_path.resolve()}\n{json_path.resolve()}")
    return 1 if statuses["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
