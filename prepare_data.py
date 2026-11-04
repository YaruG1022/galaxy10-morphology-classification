"""Galaxy10 data preparation v1. Authored with Codex assistance.

Raw HDF5 is opened read-only. Metadata indices are ORIGINAL zero-based rows.
Run --scan-only first, review candidates, then run --finalize.
Only numpy, scipy, Pillow and h5py are required. No models are trained.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np
from PIL import Image
import PIL
import scipy
from scipy.fft import dctn
from scipy.spatial import cKDTree

EXPECTED_MD5 = "a920094d5e470e3705f4691c0d6dff54"
RATIOS = np.array([0.70, 0.15, 0.15])
SPLITS = ("train", "validation", "test")


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def variants(gray):
    return [np.rot90(gray, k) for k in range(4)] + [np.rot90(np.fliplr(gray), k) for k in range(4)]


def hash_gray(gray):
    # Standard 64-bit DCT pHash, median excludes DC; grayscale dHash is a secondary diagnostic.
    low = dctn(gray.astype(float), norm="ortho")[:8, :8]
    bits = low > np.median(low.ravel()[1:])
    p = int.from_bytes(np.packbits(bits).tobytes(), "big")
    small = np.asarray(Image.fromarray(np.ascontiguousarray(gray)).resize((9, 8), Image.Resampling.LANCZOS))
    d = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    return p, d


class BKTree:
    """Exact Hamming-radius lookup, retaining every image with an identical hash."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if not distance:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def sky_pairs(ra, dec, arcsec):
    good = np.flatnonzero(np.isfinite(ra) & np.isfinite(dec))
    a, d = np.deg2rad(ra[good]), np.deg2rad(dec[good])
    xyz = np.column_stack((np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)))
    chord = 2 * np.sin(np.deg2rad(arcsec / 3600) / 2)
    pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    result = {}
    for x, y in pairs:
        i, j = sorted((int(good[x]), int(good[y])))
        separation = np.rad2deg(2*np.arcsin(min(1., np.linalg.norm(xyz[x]-xyz[y])/2)))*3600
        result[(i, j)] = float(separation)
    return result


def refine_candidates(out, thumbs):
    # pHash alone also matches smooth but different galaxies. Require strong
    # aligned thumbnail agreement for hash-only matches; always keep sky/exact flags.
    initial = out / "initial_screen_matches.csv"
    if not initial.exists():
        initial.write_bytes((out / "duplicate_candidates.csv").read_bytes())
    with initial.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    retained = []
    assessed = []
    for row in rows:
        i, j = int(row["index_a"]), int(row["index_b"])
        a = thumbs[i].astype(float)
        b = variants(thumbs[j])[int(row["orientation_b"])].astype(float)
        ac, bc = a-a.mean(), b-b.mean()
        denominator = float(np.linalg.norm(ac)*np.linalg.norm(bc))
        correlation = float(np.sum(ac*bc)/denominator) if denominator else 0.
        rmse = float(np.sqrt(np.mean((a-b)**2)))
        direct = "sky" in row["source"] or "exact" in row["source"]
        strong = correlation >= .995 and rmse <= 8 and int(row["dhash_distance"]) <= 8
        row.update(thumbnail_correlation=correlation, thumbnail_rmse=rmse,
                   retained_candidate=int(direct or strong))
        assessed.append(row)
        if direct or strong:
            retained.append(row)
    fields = list(assessed[0]) if assessed else ["index_a", "index_b", "exact_pixels", "sky_arcsec", "phash_distance", "dhash_distance", "orientation_b", "source", "thumbnail_correlation", "thumbnail_rmse", "retained_candidate"]
    write_csv(out / "image_screen_assessment.csv", assessed, fields)
    write_csv(out / "duplicate_candidates.csv", retained, fields)
    info = json.loads((out / "scan.json").read_text(encoding="utf-8"))
    info["initial_screen_pair_count"] = len(rows)
    info["candidate_pair_count"] = len(retained)
    info["thresholds"].update(hash_only_min_thumbnail_correlation=.995,
                               hash_only_max_thumbnail_rmse=8,
                               hash_only_max_dhash_distance=8)
    info["limitations"].append("Hash-only matches must also pass aligned-thumbnail correlation, RMSE and dHash thresholds; weak hash-only matches remain recorded but are not grouped. These screening thresholds are engineering choices, not calibrated duplicate probabilities.")
    dump_json(out / "scan.json", info)
    print(f"Refined candidate pairs: {len(retained)} from {len(rows)} initial matches", flush=True)


def scan(args):
    out = args.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing preparation: {out}")
    md5 = digest_file(args.data, "md5")
    if md5 != EXPECTED_MD5:
        raise ValueError("Raw-file checksum does not match the verified official release")
    out.mkdir(parents=True)
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as f:
        n = f["images"].shape[0]
        labels = f["ans"][:]
        metadata = {k: f[k][:] for k in ("ra", "dec", "redshift", "pxscale")}
        if f["images"].shape != (n, 256, 256, 3) or any(x.shape != (n,) for x in [labels, *metadata.values()]):
            raise ValueError("Unexpected dataset structure")
        if set(np.unique(labels).tolist()) != set(range(10)):
            raise ValueError("Unexpected labels")
        thumbs = np.empty((n, 32, 32), dtype=np.uint8)
        exact = {}
        exact_pairs, constant = set(), []
        # Align reads to the HDF5 first-axis chunk to avoid repeatedly decompressing
        # the same 553-row tiles in this release (about 104 MiB per aligned batch).
        batch_size = f["images"].chunks[0] if f["images"].chunks else 64
        for start in range(0, n, batch_size):
            batch = f["images"][start:start+batch_size]
            for offset, img in enumerate(batch):
                i = start + offset
                if img.min() == img.max():
                    constant.append(i)
                key = hashlib.sha256(img.tobytes()).hexdigest()
                if key in exact:
                    exact_pairs.add((exact[key], i))
                else:
                    exact[key] = i
                thumbs[i] = np.asarray(Image.fromarray(img).convert("L").resize((32, 32), Image.Resampling.LANCZOS))
            if (start // batch_size) % 4 == 0 or start+len(batch) == n:
                print(f"Reading images: {start+len(batch)}/{n}", flush=True)
    np.savez_compressed(cache / "scan_arrays.npz", thumbnails=thumbs, labels=labels, **metadata)
    tree = BKTree()
    hashes = []
    perceptual = {}
    excluded = set(constant)
    for i, gray in enumerate(thumbs):
        ph, dh = zip(*(hash_gray(v) for v in variants(gray)))
        hashes.append((ph, dh))
        if i in excluded:
            continue
        for orientation, p in enumerate(ph):
            for j, distance in tree.query(p, args.phash_distance):
                d_distance = (dh[orientation] ^ hashes[j][1][0]).bit_count()
                pair = (j, i)
                candidate = (distance, d_distance, orientation)
                if pair not in perceptual or candidate < perceptual[pair]:
                    perceptual[pair] = candidate
        tree.add(ph[0], i)
        if i % 2048 == 0:
            print(f"Perceptual-hash screening: {i}/{n}; pairs={len(perceptual)}", flush=True)
    close = sky_pairs(metadata["ra"], metadata["dec"], args.sky_arcsec)
    all_pairs = sorted(set(close) | set(perceptual) | exact_pairs)
    rows = []
    for i, j in all_pairs:
        if i in excluded or j in excluded:
            continue
        best = min(((hashes[i][0][0] ^ p).bit_count(),
                    (hashes[i][1][0] ^ d).bit_count(), k)
                   for k, (p, d) in enumerate(zip(*hashes[j])))
        rows.append({"index_a": i, "index_b": j,
                     "exact_pixels": int((i, j) in exact_pairs),
                     "sky_arcsec": close.get((i, j), ""),
                     "phash_distance": best[0], "dhash_distance": best[1],
                     "orientation_b": best[2],
                     "source": "+".join(s for s, yes in [("sky", (i,j) in close), ("phash", (i,j) in perceptual), ("exact", (i,j) in exact_pairs)] if yes)})
    fields = ["index_a", "index_b", "exact_pixels", "sky_arcsec", "phash_distance", "dhash_distance", "orientation_b", "source"]
    write_csv(out / "duplicate_candidates.csv", rows, fields)
    dump_json(out / "scan.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": args.data.name, "raw_md5": md5, "raw_size_bytes": args.data.stat().st_size,
        "source": "https://zenodo.org/records/10845026", "sample_count": n,
        "constant_image_indices": constant, "candidate_pair_count": len(rows),
        "thresholds": {"sky_arcsec": args.sky_arcsec, "phash_hamming_radius": args.phash_distance,
                       "hash_bits": 64, "orientations": 8, "thumbnail": [32,32]},
        "cache_arrays_sha256": digest_file(cache / "scan_arrays.npz"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "Pillow": PIL.__version__, "h5py": h5py.__version__},
        "limitations": ["Finite thresholds cannot prove that all near-duplicates have been found.",
                        "pHash can match different smooth galaxies; candidates are not automatically deleted.",
                        "Orientation screening covers rotations by multiples of 90 degrees and flips; it is not invariant to arbitrary crops."]})
    dump_json(out / "review_decisions.json", {
        "reviewer": "pending",
        "exclude_indices": {str(i): "constant image: no spatial morphology information" for i in constant},
        "pair_decisions": [],
        "default_pair_action": "retain_both_and_group",
        "metadata_policy": "Keep missing, zero, and negative redshift unchanged; flag for separate metadata analyses. Main models use images only."
    })
    refine_candidates(out, thumbs)
    print(f"Scan complete; constant images={constant}")


def split_groups(labels, kept, groups, seed):
    rng = np.random.default_rng(seed)
    targets = np.zeros((10, 3), dtype=int)
    for c in range(10):
        raw = RATIOS * sum(labels[i] == c for i in kept)
        targets[c] = np.floor(raw).astype(int)
        remaining = int(round(raw.sum())) - targets[c].sum()
        for col in np.argsort(-(raw - targets[c]), kind="stable")[:remaining]:
            targets[c, col] += 1
    remaining = targets.copy()
    assignment = {}
    multiple = [g for g in groups if len(g) > 1]
    rng.shuffle(multiple)
    multiple.sort(key=len, reverse=True)
    for group in multiple:
        vector = np.bincount(labels[group], minlength=10)
        feasible = np.flatnonzero(np.all(remaining >= vector[:,None], axis=0))
        if not len(feasible):
            raise ValueError("A duplicate group cannot fit stratified quotas; change allocation method, not the seed after model evaluation")
        probabilities = RATIOS[feasible] / RATIOS[feasible].sum()
        col = int(rng.choice(feasible, p=probabilities))
        remaining[:,col] -= vector
        assignment.update({i:SPLITS[col] for i in group})
    singleton = np.array([g[0] for g in groups if len(g)==1], dtype=int)
    for c in range(10):
        indices = singleton[labels[singleton] == c]
        rng.shuffle(indices)
        if len(indices) != remaining[c].sum():
            raise AssertionError("Quota mismatch")
        position=0
        for col, name in enumerate(SPLITS):
            count=int(remaining[c,col])
            assignment.update({int(i):name for i in indices[position:position+count]})
            position += count
    assert len(assignment)==len(kept)
    return assignment, targets


def finalize(args):
    out=args.output
    if (out / "summary.json").exists():
        raise FileExistsError("Split already frozen. Use a new version directory for any change.")
    scan_info=json.loads((out / "scan.json").read_text(encoding="utf-8"))
    if digest_file(args.data,"md5") != scan_info["raw_md5"]:
        raise ValueError("Source changed since scan")
    if digest_file(args.cache / "scan_arrays.npz") != scan_info["cache_arrays_sha256"]:
        raise ValueError("Cached arrays changed since scan")
    arrays=np.load(args.cache / "scan_arrays.npz", allow_pickle=False)
    labels=arrays["labels"]
    review=json.loads((out / "review_decisions.json").read_text(encoding="utf-8"))
    if review["reviewer"]=="pending":
        raise ValueError("Record review decisions before finalizing")
    exclusions={int(k):v for k,v in review["exclude_indices"].items()}
    if not set(scan_info["constant_image_indices"]).issubset(exclusions):
        raise ValueError("Every constant image must have an exclusion decision")
    n=len(labels)
    if any(i<0 or i>=n for i in exclusions):
        raise ValueError("Invalid excluded original index")
    kept=[i for i in range(n) if i not in exclusions]
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        parent[max(a,b)]=min(a,b)
    with (out / "duplicate_candidates.csv").open(encoding="utf-8",newline="") as f:
        pairs=[(int(r["index_a"]),int(r["index_b"])) for r in csv.DictReader(f)]
    for i,j in pairs:
        if i in exclusions or j in exclusions:
            continue
        # Even visually distinct close/similar candidates stay grouped conservatively.
        union(i,j)
    grouped={}
    for i in kept:
        grouped.setdefault(find(i),[]).append(i)
    groups=list(grouped.values())
    assignment,targets=split_groups(labels,kept,groups,args.seed)
    rows=[]
    z=arrays["redshift"]
    for i in range(n):
        rows.append({"original_index":i,"label":int(labels[i]),"included":int(i not in exclusions),
                     "exclusion_reason":exclusions.get(i,""),"group_id":find(i) if i not in exclusions else "",
                     "split":assignment.get(i,"excluded"),"redshift_missing":int(not np.isfinite(z[i])),
                     "redshift_negative":int(np.isfinite(z[i]) and z[i]<0),"redshift_zero":int(z[i]==0)})
    manifest=out / "sample_manifest.csv"
    write_csv(manifest,rows,list(rows[0]))
    actual={}
    for name in SPLITS:
        indices=sorted(i for i,s in assignment.items() if s==name)
        dump_json(out / f"{name}_indices.json",indices)
        actual[name]={"count":len(indices),"fraction":len(indices)/len(kept),
                      "class_counts":np.bincount(labels[indices],minlength=10).tolist()}
    flat=[i for name in SPLITS for i in json.loads((out / f"{name}_indices.json").read_text())]
    assert len(flat)==len(set(flat))==len(kept) and set(flat)==set(kept)
    assert all(assignment[i]==assignment[j] for i,j in pairs if i in assignment and j in assignment)
    assert all(len({assignment[i] for i in g})==1 for g in groups)
    for col,name in enumerate(SPLITS):
        assert actual[name]["class_counts"]==targets[:,col].tolist()
    file_hashes={p.name:digest_file(p) for p in out.iterdir() if p.is_file()}
    summary={"status":"data_preparation_complete_with_documented_limits", "seed":args.seed,
             "raw_count":n,"included_count":len(kept),"excluded_count":len(exclusions),
             "included_class_counts":np.bincount(labels[kept],minlength=10).tolist(),
             "split":actual,"target_ratios":RATIOS.tolist(),
             "allocation":"Per-class largest-remainder integer quotas; group members assigned together, then singletons fill quotas. Ties prefer train, validation, test.",
             "candidate_pairs":len(pairs),"protected_multi_sample_groups":sum(len(g)>1 for g in groups),
             "largest_group":max(map(len,groups)),"candidate_pairs_crossing_splits":0,
             "original_index_convention":"Zero-based row number in verified raw HDF5; never renumber after exclusions",
             "raw_md5":scan_info["raw_md5"],"manifest_sha256":digest_file(manifest),
             "artifact_sha256":file_hashes,"reviewer":review["reviewer"],
             "script_sha256":digest_file(Path(__file__)),
             "remaining_limits":scan_info["limitations"],
             "no_training_statistics_fitted":True}
    dump_json(out / "summary.json",summary)
    print(json.dumps(summary,indent=2))


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=root / "data/raw/Galaxy10_DECals_NoDuplicated.h5")
    parser.add_argument("--output",type=Path,default=root / "metadata/data_v1")
    parser.add_argument("--cache",type=Path,default=root / "data/processed/preparation_v1")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--sky-arcsec",type=float,default=5.)
    parser.add_argument("--phash-distance",type=int,default=4)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan-only",action="store_true")
    mode.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    if args.sky_arcsec<=0 or not 0<=args.phash_distance<=64:
        parser.error("Invalid screening thresholds")
    if args.scan_only:
        scan(args)
    else:
        finalize(args)


if __name__=="__main__":
    main()
