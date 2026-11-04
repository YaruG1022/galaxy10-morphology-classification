"""Independently verify frozen Galaxy10 manifests without regenerating a split."""
import csv
import hashlib
import json
from pathlib import Path
import argparse

import h5py
import numpy as np


def digest(path, algorithm="sha256"):
    h=hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def validate(raw, directory):
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert digest(raw,"md5")==summary["raw_md5"], "Raw file changed"
    assert digest(directory.parent.parent/"prepare_data.py")==summary["script_sha256"], "Preparation script changed after freezing"
    for filename, expected in summary["artifact_sha256"].items():
        assert digest(directory/filename)==expected, f"Artifact changed: {filename}"
    with h5py.File(raw,"r") as f:
        labels=f["ans"][:]
        n=len(labels)
        assert len(f["images"])==n==summary["raw_count"]
    with (directory/"sample_manifest.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==n
    indices=[int(r["original_index"]) for r in rows]
    assert indices==list(range(n)), "Missing, repeated or reordered original indices"
    assert all(int(r["label"])==int(labels[i]) for i,r in enumerate(rows))
    kept={int(r["original_index"]) for r in rows if r["included"]=="1"}
    excluded=set(range(n))-kept
    assert len(kept)==summary["included_count"]
    assert len(excluded)==summary["excluded_count"]
    decisions=json.loads((directory/"review_decisions.json").read_text(encoding="utf-8"))
    assert excluded=={int(i) for i in decisions["exclude_indices"]}
    assignment={}
    for split in ("train","validation","test"):
        values=json.loads((directory/f"{split}_indices.json").read_text(encoding="utf-8"))
        assert len(values)==len(set(values))
        assert all(i in kept and i not in assignment for i in values), "Overlap/excluded row in splits"
        assignment.update({i:split for i in values})
        assert len(values)==summary["split"][split]["count"]
        assert np.bincount(labels[values],minlength=10).tolist()==summary["split"][split]["class_counts"]
    assert set(assignment)==kept
    assert all(r["split"]==assignment.get(i,"excluded") for i,r in enumerate(rows))
    group_split={}
    for i in sorted(kept):
        group=rows[i]["group_id"]
        assert group_split.setdefault(group,assignment[i])==assignment[i], "Group crosses partitions"
    with (directory/"duplicate_candidates.csv").open(encoding="utf-8",newline="") as f:
        pairs=list(csv.DictReader(f))
    for pair in pairs:
        i,j=int(pair["index_a"]),int(pair["index_b"])
        if i in kept and j in kept:
            assert assignment[i]==assignment[j], "Candidate pair crosses partitions"
    for i in excluded:
        assert rows[i]["exclusion_reason"]
    return {"raw_count":n,"retained":len(kept),"excluded":len(excluded),
            "split_counts":{s:summary["split"][s]["count"] for s in summary["split"]},
            "candidate_pairs_checked":len(pairs),"overlap":0,"candidate_leakage":0,
            "all_recorded_hashes_match":True}


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=root/"data/raw/Galaxy10_DECals_NoDuplicated.h5")
    p.add_argument("--metadata",type=Path,default=root/"metadata/data_v1")
    args=p.parse_args()
    print(json.dumps(validate(args.data,args.metadata),indent=2))
