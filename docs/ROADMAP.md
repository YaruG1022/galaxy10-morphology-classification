# Research roadmap

This schedule is carried forward from the October 7, 2026 proposal. Local data preparation v1 and training-only data-level exploration were completed on November 3, 2026. Handcrafted features, training-only preprocessing and handcrafted-feature UMAP are now complete. Classifier training, pretrained-representation exploration and performance results remain pending.

| Planned dates (2026) | Stage | Expected deliverable | Initial lead |
| --- | --- | --- | --- |
| Oct 8-14 | Repository, compute preparation, data integrity, duplicate checks, shared split | Verified data and saved split | Yaru Gao; Zijian Zhang for compute |
| Oct 15-21 | Exploratory analysis and hand-crafted features | Exploration notebook and feature table | Yaru Gao |
| Oct 22-28 | Classical tuning and first scratch-CNN runs | Baseline comparison | Zhongyan Zhang; Zijian Zhang |
| Oct 29-Nov 4 | First pretrained-model runs and shared evaluation | Initial ten-model comparison | Zijian Zhang; Zhongyan Zhang |
| Nov 5-18 | Remaining seeds, imbalance experiments, H1, interpretability | Comparison tables and error analysis | All members |
| Nov 19-29 | Final comparisons and rerun buffer | Frozen research results | All members |
| Nov 30-Dec 6 | Report, slides, and rehearsal | Draft research report and presentation | All members |
| Dec 7-14 | Present findings | Research presentation | All members |
| Dec 18 | Package final materials | Report, code, slides, and reproducibility records | All members |

## Readiness checkpoints

- **Initial review, Nov 4:** verified data and shared split; exploratory figures with interpretations; preliminary model comparisons; completed contributions and remaining tasks.
- **Results freeze, Nov 29:** complete seed runs, documented model comparisons, limitations, and error analysis.
- **Research package, Dec 18:** consistent report, code, presentation, data-access instructions, and experiment records.

## Current status

- [x] Repository and main branch.
- [x] Planning documents, ignore rules, and collaboration templates.
- [x] Local data-only dependencies and verified data acquisition.
- [x] Data inspection, documented visual duplicate review, cleaning manifest, and frozen v1 split.
- [ ] Kamiak environment, data transfer, and compute access verification.
- [x] Training-only data-level exploration: seven reviewed figures, interpretations, aggregate tables, executed notebook, and independent checks.
- [x] Handcrafted feature v1 matrices, documented proxies and masks, train-only preprocessing, and training handcrafted-feature UMAP.
- [x] Feature row/provenance/fit validation, targeted regression checks and an executed feature-audit notebook.
- [ ] Pretrained-representation UMAP with the later pretrained-model module.
- [ ] Models, evaluation, and experiments.
- [ ] Results, interpretation, report, and slides.

Data preparation and exploratory scripts, pinned data/EDA dependencies, frozen metadata, and an executed exploration notebook are present. See [the EDA record](eda/README.md). Raw data and caches remain local and ignored. Handcrafted features and fitted unsupervised preprocessing are present; see [the feature record](features/README.md). No research classifier, classifier checkpoint, or model-performance result is present.
