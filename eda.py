"""Training-only Galaxy10 exploratory analysis, with frozen v1 split provenance.

Run: python eda.py
All documents/figures use English. Authored with Codex assistance.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import textwrap

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from validate_data_preparation import validate

CLASSES = ["Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
           "Cigar-Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
           "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"]
SEED = 20261103
CREDIT = "Galaxy10 DECaLS: Leung & Bovy / Zenodo 10845026; images from DESI Legacy Imaging Surveys"
NAMES = ["class_balance", "training_examples_classes_0_4", "training_examples_classes_5_9",
         "training_class_means", "training_intensity", "training_channel_differences", "training_metadata"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def describe(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {k: None for k in ["min", "q05", "q25", "median", "q75", "q95", "max"]}
    return dict(zip(["min", "q05", "q25", "median", "q75", "q95", "max"],
                    map(float, np.quantile(finite, [0, .05, .25, .5, .75, .95, 1]))))


def select_examples(train, labels, seed=SEED, per_class=4):
    rng = np.random.default_rng(seed)
    return {str(c): sorted(map(int, rng.choice(train[labels[train] == c], size=per_class, replace=False)))
            for c in range(10)}


def image_statistics(image):
    channel_mean = image.mean(axis=(0, 1), dtype=np.float64) / 255.
    gray = image.mean(axis=2, dtype=np.float64) / 255.
    return np.r_[channel_mean, gray.mean(), gray.std(ddof=0)]


def collect_training(raw, train, labels, examples):
    """Process only selected training rows; contiguous HDF5 buffers are filtered immediately."""
    n = len(labels)
    if len(set(map(int, train))) != len(train) or np.any(train < 0) or np.any(train >= n):
        raise ValueError("Training indices are repeated or out of range")
    train = np.sort(np.asarray(train, dtype=int))
    positions = {int(i): k for k, i in enumerate(train)}
    wanted = {i for values in examples.values() for i in values}
    if not wanted.issubset(positions):
        raise ValueError("Gallery includes a non-training row")
    stats = np.empty((len(train), 5), dtype=np.float64)
    sums = np.zeros((10, 256, 256, 3), dtype=np.float64)
    counts = np.zeros(10, dtype=int)
    pictures = {}
    with h5py.File(raw, "r") as f:
        images = f["images"]
        step = images.chunks[0] if images.chunks else 64
        for start in range(0, n, step):
            subset = train[(train >= start) & (train < start+step)]
            if not len(subset):
                continue
            # The release tiles 553 rows per compressed block. Only the selected
            # training rows enter image statistics, examples, or class averages.
            block = images[start:start+step]
            for i in subset:
                img = block[int(i)-start]
                c = int(labels[i])
                stats[positions[int(i)]] = image_statistics(img)
                sums[c] += img
                counts[c] += 1
                if int(i) in wanted:
                    pictures[int(i)] = img.copy()
            if (start//step) % 4 == 0 or start+step >= n:
                print(f"Training images summarized: {int(counts.sum())}/{len(train)}", flush=True)
        metadata = {k: f[k][:][train] for k in ["redshift", "pxscale"]}
    assert int(counts.sum()) == len(train) and np.isfinite(stats).all()
    assert set(pictures) == wanted
    return train, stats, sums / counts[:, None, None, None] / 255., counts, pictures, metadata


def styling():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.hashsalt": "galaxy10-eda-v1"})


def figure_save(fig, out, name, vector=False):
    fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight",
                metadata={"Software": "Galaxy10 reproducible training-only EDA"})
    if vector:
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def boxes(ax, values, title, ylabel, colors):
    bp = ax.boxplot(values, patch_artist=True, showfliers=False, whis=(5, 95),
                    tick_labels=[str(i) for i in range(10)], widths=.62,
                    medianprops={"color": "black", "linewidth": 1.4})
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.72)
    ax.set(title=title, ylabel=ylabel, xlabel="Class ID")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def render_figures(out, rows, train, labels, stats, means, counts, pictures, examples, meta):
    styling()
    colors = plt.get_cmap("tab10")(np.arange(10))
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    totals = np.array([r["retained"] for r in rows])
    bars = ax.barh(np.arange(10), totals, color=colors)
    for bar, value in zip(bars, totals):
        ax.text(value+25, bar.get_y()+bar.get_height()/2,
                f"{value:,} ({value/totals.sum():.1%})", va="center", fontsize=10)
    ax.set_yticks(np.arange(10), [f"{i}  {name}" for i, name in enumerate(CLASSES)])
    ax.invert_yaxis()
    ax.set(xlim=(0, max(totals)*1.28), xlabel="Retained image count",
           title="Retained class balance | 17,655 images")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    figure_save(fig, out, "class_balance", vector=True)

    for start in [0, 5]:
        fig, axes = plt.subplots(5, 4, figsize=(11, 12.5), layout="constrained")
        for row, c in enumerate(range(start, start+5)):
            for col, i in enumerate(examples[str(c)]):
                ax = axes[row, col]
                ax.imshow(pictures[i], interpolation="nearest", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"Original row {i}", fontsize=9, fontweight="normal")
                if col == 0:
                    ax.set_ylabel(f"{c}: " + "\n".join(textwrap.wrap(CLASSES[c], 19)), fontsize=10)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle(f"Training examples | classes {start}-{start+4}\nFour seeded samples per class; seed {SEED}", fontsize=14)
        fig.supxlabel(CREDIT, fontsize=7)
        figure_save(fig, out, f"training_examples_classes_{start}_{start+4}")

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.4), layout="constrained")
    for c, ax in enumerate(axes.flat):
        ax.imshow(means[c], interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{c}: {CLASSES[c]}\nn={counts[c]:,}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle("Mean stored-color image per class | training only", fontsize=14)
    fig.supxlabel("Original 256 x 256 pixels; arithmetic average; no orientation alignment or contrast rescaling.\n"+CREDIT, fontsize=8)
    figure_save(fig, out, "training_class_means")

    y = labels[train]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    boxes(axes[0], [stats[y==c,3] for c in range(10)], "Mean stored intensity", "Pixel-value average / 255", colors)
    boxes(axes[1], [stats[y==c,4] for c in range(10)], "Within-image spatial variation", "Std. of channel-averaged pixels / 255", colors)
    fig.suptitle("Training-image pixel distributions", fontsize=14)
    fig.supxlabel("Boxes: 25th-75th percentiles; line: median; whiskers: 5th-95th percentiles. Tails omitted from drawing, not computation.", fontsize=8)
    figure_save(fig, out, "training_intensity", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, a, b in [(axes[0],0,1),(axes[1],1,2)]:
        boxes(ax, [stats[y==c,a]-stats[y==c,b] for c in range(10)],
              f"Stored-channel mean {a} minus {b}", "Difference in normalized stored pixel values", colors)
        ax.axhline(0, color="gray", linewidth=.8, linestyle="--")
    fig.suptitle("Training-image stored-channel differences", fontsize=14)
    fig.supxlabel("Display-composite diagnostics, not calibrated astronomical color indices. Boxes/whiskers: 25-75% / 5-95%.", fontsize=8)
    figure_save(fig, out, "training_channel_differences", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    z = meta["redshift"]
    positive = np.isfinite(z) & (z>0)
    boxes(axes[0], [z[(y==c)&positive] for c in range(10)],
          "Positive finite redshift", "Redshift (log scale)", colors)
    axes[0].set_yscale("log")
    scales = np.unique(meta["pxscale"])
    bottom = np.zeros(10)
    for scale, color in zip(scales, ["#356894", "#d4a46a", "#92b4a7", "#806b95"]):
        fractions = np.array([np.mean(meta["pxscale"][y==c]==scale) for c in range(10)])
        axes[1].bar(np.arange(10), fractions*100, bottom=bottom*100, color=color,
                    label=f"{scale:g} arcsec/pixel")
        bottom += fractions
    axes[1].set(xticks=np.arange(10), xlabel="Class ID", ylabel="Within-class images (%)",
                title="Image pixel scale", ylim=(0,100))
    axes[1].legend(fontsize=9)
    fig.suptitle("Training metadata | descriptive context, excluded from main model inputs", fontsize=13)
    fig.supxlabel("Redshift panel omits missing/nonpositive values only for this log-scale plot; images remain in the dataset.\nBoxes/whiskers: 25-75% / 5-95%; all omissions are counted in the tables.", fontsize=8)
    figure_save(fig, out, "training_metadata", vector=True)


def write_report(out, summary, class_rows, pixel_rows, meta_rows):
    n = summary["training_count"]
    smallest = min(class_rows, key=lambda r:r["retained"])
    largest = max(class_rows, key=lambda r:r["retained"])
    dark = min(pixel_rows, key=lambda r:r["intensity_median"])
    bright = max(pixel_rows, key=lambda r:r["intensity_median"])
    low_z = min(meta_rows, key=lambda r:r["positive_redshift_median"])
    high_z = max(meta_rows, key=lambda r:r["positive_redshift_median"])
    lines = ["# Galaxy10 DECaLS: exploratory data analysis", "",
             "This record describes the prepared v1 dataset and its training images. It contains no trained-model results.", "",
             "## Questions and scope", "",
             "How uneven are the classes? What do the images look like within each class? Do stored image values and metadata differ between classes? These checks inform later experiments; they do not establish classification performance or support the CNN-versus-transformer hypothesis.", "",
             f"The frozen dataset contains 17,655 retained images: 12,359 training, 2,650 validation, and 2,646 test. Every image-derived figure and numeric distribution below uses only the {n:,} training rows. Full-dataset class totals come from the frozen manifest; validation/test images are not used for exploratory statistics or gallery selection.", "",
             "## Class balance", "",
             f"The largest retained class is {largest['class_id']} ({largest['class_name']}, {largest['retained']:,} images); the smallest is {smallest['class_id']} ({smallest['class_name']}, {smallest['retained']:,}). Their size ratio is {summary['retained_imbalance_ratio']:.2f}:1. Class 4 accounts for only {smallest['retained']/17655:.2%} of retained images, so overall accuracy alone can hide poor minority-class performance. The planned macro F1 and per-class recall remain appropriate reporting choices.", "",
             "![Retained class balance](figures/class_balance.png)", "",
             "| ID | Class | Retained | Train | Validation | Test |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in class_rows:
        lines.append(f"| {r['class_id']} | {r['class_name']} | {r['retained']} | {r['train']} | {r['validation']} | {r['test']} |")
    lines += ["", "Counts sum to the frozen split exactly. Integer allocation causes small differences from exact 70/15/15 fractions; the split has not been changed for this analysis.", "",
              "## Within-class examples", "",
              f"Four images per class were sampled without replacement from training using seed {SEED}. Titles retain the original zero-based HDF5 row numbers. Sampling was not based on image quality, prediction difficulty, or model outputs. The gallery is a small sample, not proof that each class is visually separable or every label is correct.", "",
              "![Training examples, classes 0-4](figures/training_examples_classes_0_4.png)", "",
              "![Training examples, classes 5-9](figures/training_examples_classes_5_9.png)", "",
              "The sampled cutouts include bright neighboring sources and targets with different apparent sizes; original rows 201, 3362 and 16009 illustrate this context. Whole-cutout intensity and channel averages therefore describe the target plus its background and neighbors. These examples do not estimate how common contamination is. No further exclusions or label changes were made after the split was frozen.", "",
              "## Average images", "",
              "Each panel is the arithmetic average of all training images in one class, at the original resolution and stored-color scale. Images are not rotated or aligned before averaging. Spiral details can therefore blur; the means summarize common spatial patterns and centering, rather than replace individual examples or demonstrate predictive features.", "",
              "![Training class means](figures/training_class_means.png)", "",
              "All ten displayed averages retain a bright central region. The mean images are much smoother than the individual examples, so an average image alone is insufficient evidence of class separability.", "",
              "## Stored pixel distributions", "",
              "Mean intensity is the average of all stored pixel-channel values divided by 255. Spatial variation is the population standard deviation of per-pixel channel averages, also divided by 255. Stored-channel differences subtract each image's normalized channel means. These are display-composite diagnostics, not calibrated fluxes, surface brightnesses, magnitudes, or physical g-r/r-z colors.", "",
              f"In this training sample, class {dark['class_id']} has the lowest median mean intensity ({dark['intensity_median']:.4f}) and class {bright['class_id']} the highest ({bright['intensity_median']:.4f}). This is a descriptive difference; acquisition, display rendering, background, pixel scale, and object properties are not isolated here. It cannot be interpreted as a causal effect of morphology.", "",
              "![Training intensity](figures/training_intensity.png)", "",
              "![Training stored-channel differences](figures/training_channel_differences.png)", "",
              "Boxes span the 25th-75th percentiles, lines mark medians, and whiskers mark the 5th-95th percentiles. Tails are hidden from the drawing only; exported tables include minima and maxima. These intervals describe variation among images, not confidence intervals for a mean.", "",
              "## Metadata context", "",
              f"Redshift counts in the training partition are: missing={summary['training_redshift_missing']}, negative={summary['training_redshift_negative']}, zero={summary['training_redshift_zero']}. These images remain included. The positive-redshift plot uses only finite values greater than zero because its vertical axis is logarithmic. The main comparison continues to exclude redshift and pixel scale from model inputs.", "",
              "![Training metadata](figures/training_metadata.png)", "",
              "| Class ID | Train n | Missing z | Negative z | Zero z | Positive-z median |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in meta_rows:
        median = r['positive_redshift_median']
        lines.append(f"| {r['class_id']} | {r['count']} | {r['redshift_missing']} | {r['redshift_negative']} | {r['redshift_zero']} | {median:.4f} |")
    lines += ["", f"Among positive finite training redshifts, class {low_z['class_id']} has the lowest median ({low_z['positive_redshift_median']:.4f}) and class {high_z['class_id']} the highest ({high_z['positive_redshift_median']:.4f}). These distributions overlap and do not by themselves explain morphology or demonstrate that metadata improve prediction.", "",
              "Redshift and pixel-scale distributions provide context for potential dataset biases. They do not establish why classes differ. Finite sentinel codes and the physical meaning of individual nonpositive redshifts have not been independently verified.", "",
              "## Reproduce and verify", "", "From the repository root, with the original file under `data/raw/`:", "", "```powershell", "python -m pip install -r requirements-eda.txt", "python eda.py", "python validate_eda.py", "```", "",
              "The executed notebook `notebooks/01_exploration.ipynb` provides a walkthrough. Machine-readable results are under `docs/eda/tables/`; provenance, sampled row IDs and output hashes are in `docs/eda/summary.json`. Per-image diagnostics and average arrays are local under ignored `data/processed/eda_v1/`.", "",
              "## Boundaries and next work", "",
              "This completes the data-level exploration module. Hand-crafted feature extraction, feature/pretrained-representation UMAP plots, model training, prediction errors and the research hypothesis require later modules. No model has been evaluated, no test-set pixels inform these descriptive results, and no normalization or dimensionality-reduction estimator has been fitted for modeling.", "",
              "The first-stage duplicate decisions were made using finite screening thresholds and Codex visual review; they do not prove that every duplicate or label error is absent. The team should independently verify the decisions and the interpretations recorded here.", "",
              "## Sources and assistance", "",
              "Data: W. Henry Leung and Jo Bovy, [Galaxy10 DECaLS, Zenodo record 10845026](https://zenodo.org/records/10845026), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The license ID was verified from the record's public API. Source images: DESI Legacy Imaging Surveys; class labels: Galaxy Zoo. Changes here include selecting and arranging training-image examples, resizing them for figure display, averaging images, and computing descriptive aggregates. See [the dataset documentation](https://astronn.readthedocs.io/en/latest/galaxy10.html) and [our preparation record](../DATA.md). Raw data are not uploaded.", "",
              "EDA scripts, figures, the notebook and this exploratory record were produced with Codex assistance. Independent team verification has not been recorded. This is a generated exploratory record, not a statement that the team independently wrote or interpreted these materials.", ""]
    with (out/'README.md').open('w',encoding='utf-8',newline='\n') as f:
        f.write('\n'.join(lines))


def run(root, refresh=False):
    raw = root / "data/raw/Galaxy10_DECals_NoDuplicated.h5"
    frozen = root / "metadata/data_v1"
    out = root / "docs/eda"
    local = root / "data/processed/eda_v1"
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    tables = out / "tables"; tables.mkdir(exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    validation = validate(raw, frozen)
    with h5py.File(raw, "r") as f:
        labels = f["ans"][:]
    split_indices = {s: np.array(json.loads((frozen/f"{s}_indices.json").read_text()), dtype=int)
                     for s in ["train", "validation", "test"]}
    train = split_indices["train"]
    examples = select_examples(train, labels)
    cache = local / "training_statistics.npz"
    cache_record = local / "cache_record.json"
    source_key = {"source_md5": "a920094d5e470e3705f4691c0d6dff54",
                  "manifest_sha256": sha(frozen / "sample_manifest.csv"),
                  "train_indices_sha256": sha(frozen / "train_indices.json"),
                  "script_sha256": sha(Path(__file__)), "gallery_seed": SEED}
    cached = False
    if cache.exists() and cache_record.exists() and not refresh:
        info = json.loads(cache_record.read_text())
        if info["source_key"] == source_key and info["cache_sha256"] == sha(cache):
            with np.load(cache, allow_pickle=False) as a:
                train, stats, means, counts = [a[k].copy() for k in ["train", "stats", "means", "counts"]]
                pictures = {int(i): img.copy() for i, img in zip(a["example_ids"], a["example_images"])}
                meta = {k:a[k].copy() for k in ["redshift", "pxscale"]}
            cached = True
    if not cached:
        train, stats, means, counts, pictures, meta = collect_training(raw, train, labels, examples)
        ids = np.array(sorted(pictures))
        np.savez_compressed(cache, train=train, stats=stats, means=means, counts=counts,
                            example_ids=ids, example_images=np.array([pictures[int(i)] for i in ids]), **meta)
        save_json(cache_record, {"source_key":source_key,"cache_sha256":sha(cache)})
    assert np.array_equal(train, split_indices["train"])
    assert np.array_equal(counts, np.bincount(labels[train], minlength=10))
    class_rows, pixel_rows, meta_rows = [], [], []
    y=labels[train]
    for c in range(10):
        row={"class_id":c,"class_name":CLASSES[c]}
        row.update({s:int(np.sum(labels[idx]==c)) for s,idx in split_indices.items()})
        row["retained"]=row["train"]+row["validation"]+row["test"]
        class_rows.append(row)
        pixel={"class_id":c,"count":int(counts[c])}
        metrics={"intensity":stats[y==c,3],"spatial_std":stats[y==c,4],
                 "channel_0_minus_1":stats[y==c,0]-stats[y==c,1],
                 "channel_1_minus_2":stats[y==c,1]-stats[y==c,2]}
        for metric, values in metrics.items():
            pixel.update({metric+"_"+k:v for k,v in describe(values).items()})
        pixel_rows.append(pixel)
        z=meta["redshift"][y==c]
        metadata={"class_id":c,"count":int(counts[c]),"redshift_missing":int((~np.isfinite(z)).sum()),
                  "redshift_negative":int((np.isfinite(z)&(z<0)).sum()),"redshift_zero":int((z==0).sum())}
        metadata.update({"positive_redshift_"+k:v for k,v in describe(z[np.isfinite(z)&(z>0)]).items()})
        for scale in np.unique(meta["pxscale"]):
            metadata[f"pxscale_{scale:g}_count"]=int(np.sum(meta["pxscale"][y==c]==scale))
        meta_rows.append(metadata)
    save_csv(tables/'class_counts.csv',class_rows)
    save_csv(tables/'training_pixel_summary.csv',pixel_rows)
    save_csv(tables/'training_metadata_summary.csv',meta_rows)
    per_image=[dict(original_index=int(i),label=int(labels[i]),channel_0_mean=float(s[0]),
                    channel_1_mean=float(s[1]),channel_2_mean=float(s[2]),
                    intensity=float(s[3]),spatial_std=float(s[4])) for i,s in zip(train,stats)]
    save_csv(local/'training_image_statistics.csv',per_image)
    render_figures(figures,class_rows,train,labels,stats,means,counts,pictures,examples,meta)
    summary={"created_at_utc":datetime.now(timezone.utc).isoformat(),"scope":"training_images_only",
             "training_count":len(train),"retained_count":sum(r['retained'] for r in class_rows),
             "gallery_seed":SEED,"example_indices":examples,
             "retained_imbalance_ratio":max(r['retained'] for r in class_rows)/min(r['retained'] for r in class_rows),
             "training_redshift_missing":int((~np.isfinite(meta['redshift'])).sum()),
             "training_redshift_negative":int((np.isfinite(meta['redshift'])&(meta['redshift']<0)).sum()),
             "training_redshift_zero":int((meta['redshift']==0).sum()),
             "class_counts":class_rows,"source_key":source_key,
             "input_validation":validation,
             "versions":{"python":platform.python_version(),"numpy":np.__version__,"h5py":h5py.__version__,"matplotlib":matplotlib.__version__},
             "statistics_definitions":{"intensity":"Mean of all stored pixel-channel values / 255",
                 "spatial_std":"Population std of per-pixel channel averages / 255",
                 "channel_differences":"Differences of normalized stored-channel means; not calibrated astronomical colors"},
             "dataset_license":"CC-BY-4.0", "license_verified_from":"https://zenodo.org/api/records/10845026",
             "test_images_analyzed":0,"model_training_performed":False,"feature_estimators_fitted":False}
    write_report(out,summary,class_rows,pixel_rows,meta_rows)
    summary['output_sha256']={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['summary.json','validation_report.json']}
    save_json(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['training_count','retained_count','retained_imbalance_ratio','training_redshift_missing','training_redshift_negative','training_redshift_zero']},indent=2))
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--refresh',action='store_true',help='Recompute training-only pixel statistics instead of reusing a verified cache')
    args=p.parse_args()
    run(args.root.resolve(),args.refresh)
