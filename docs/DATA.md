# Dataset plan

**Status: local data preparation v1 completed on November 3, 2026.** The raw file is unchanged. A reviewed sample manifest and fixed partition indices are saved under `metadata/data_v1/`. No trained model or performance result is included.

## Source

- Dataset: Galaxy10 DECaLS, compiled by Henry Leung and Jo Bovy.
- Planned file: `Galaxy10_DECals_NoDuplicated.h5`.
- Download record: https://zenodo.org/records/10845026
- Dataset documentation: https://astronn.readthedocs.io/en/latest/galaxy10.html
- Dataset DOI cited in the proposal: https://doi.org/10.5281/zenodo.10845025
- Images: DESI Legacy Imaging Surveys; labels: Galaxy Zoo volunteer classifications.

The proposal identifies the dataset license as Creative Commons Attribution 4.0 (CC BY 4.0). Verify the selected file's record and license before reuse, and cite the dataset creators and record. This dataset license does not establish a software license for the repository.

## Verified fields

The local file has 17,675 records. `images` has shape `(17675, 256, 256, 3)` and dtype `uint8`; `ans` has shape `(17675,)` and dtype `uint8`; the four metadata arrays have shape `(17675,)` and dtype `float64`. Labels are exactly 0 through 9.

| Field | Planned use |
| --- | --- |
| `images` | 256 x 256 color image input; g, r, and z band composites |
| `ans` | Class label, 0 through 9 |
| `ra`, `dec` | Sky coordinates for duplicate checks |
| `redshift` | Error analysis and a separate metadata ablation |
| `pxscale` | Zoom-level analysis and a separate metadata ablation |

The original release contains 17,736 images. Our verified NoDuplicated file contains 17,675; data preparation v1 retains 17,655. These are three different counts, not interchangeable versions of the same count.

## Class labels

| ID | Class |
| --- | --- |
| 0 | Disturbed |
| 1 | Merging |
| 2 | Round Smooth |
| 3 | In-between Round Smooth |
| 4 | Cigar-Shaped Smooth |
| 5 | Barred Spiral |
| 6 | Unbarred Tight Spiral |
| 7 | Unbarred Loose Spiral |
| 8 | Edge-on without Bulge |
| 9 | Edge-on with Bulge |

## Completed local preparation v1

Source: `Galaxy10_DECals_NoDuplicated.h5` from the Zenodo record above. Verified size: 2,726,423,918 bytes. MD5: `a920094d5e470e3705f4691c0d6dff54`. The source file has not been edited. The actual download timestamp was not recorded reliably; the preparation date is not a download date.

### Cleaning decisions

- Keep images whose redshift is missing, zero, or negative; do not fill or take absolute values. The raw array has 92 NaNs, two zeros and one negative value. Missing/nonpositive redshifts require separate decisions if metadata are later analyzed. Main comparisons use image inputs only.
- Exclude original row 11475: every image element has value 32, so no spatial morphology can be measured.
- No exact pixel duplicates were found by SHA-256 hashing all images.
- Sky screening within 5 arcseconds found 18 candidate edges forming 17 groups. All 18 pairs were visually inspected by Codex using full-image review sheets. Matching galaxy structures and background sources indicated redundant, shifted overlapping cutouts.
- For the 16 groups with consistent labels, keep the lowest original index and exclude 17 redundant rows (one group contains three views).
- Exclude both rows 1006 and 12442 in the overlapping-view group with conflicting labels 0 and 7. No label was selected or changed.
- Total exclusions: 1 constant image + 17 redundant views + 2 label-conflicting views = 20. Retained: 17,655.

These are Codex-assisted preparation decisions, not an independently confirmed team review. Evidence and per-group reasons are in `review_decisions.json`; the three visual review sheets are in ignored local `data/processed/preparation_v1/review_images/`.

### Near-duplicate screening method and limits

Images were converted to 32-by-32 grayscale thumbnails for 64-bit DCT perceptual hashing, with eight right-angle rotation/reflection variants. The initial union of sky neighbors, pHash Hamming distance <=4, and exact-pixel matches contained 21,504 pairs. Smooth galaxies can share coarse hashes.

For hash-only matches, secondary screening required aligned-thumbnail correlation >=0.995, RMSE <=8 pixel levels, and dHash distance <=8. Sky-neighbor or exact-pixel flags were always kept for review. All initial matches and secondary assessments are retained in CSV files; 18 pairs remained, all from sky proximity. No image was removed solely because its hash matched.

The thresholds are documented engineering choices, not calibrated duplicate probabilities. This procedure cannot prove the absence of duplicates outside these thresholds, arbitrary crops, or different fields of view. Any newly identified duplicate affecting the split requires a new version and a rerun of affected experiments.

Sky positions are mapped to three-dimensional unit vectors, then searched by chord distance; this handles RA wrap-around and declination correctly. See the [SciPy query_pairs documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html). Group separation follows the principle that related observations must not cross partitions; see [scikit-learn's grouped and stratified splitting guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

### Frozen shared split

Seed: **42**. Original row indices are **zero-based positions in the unchanged raw HDF5**, never positions in a renumbered filtered array. Integer per-class quotas are allocated by largest remainders, then shuffled deterministically. Related retained candidates would be kept together; all reviewed redundant groups in v1 now have at most one retained view.

| Class | Raw | Retained | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| 0 | 1081 | 1080 | 756 | 162 | 162 |
| 1 | 1853 | 1844 | 1291 | 277 | 276 |
| 2 | 2640 | 2640 | 1848 | 396 | 396 |
| 3 | 2021 | 2021 | 1415 | 303 | 303 |
| 4 | 333 | 333 | 233 | 50 | 50 |
| 5 | 2040 | 2039 | 1427 | 306 | 306 |
| 6 | 1823 | 1821 | 1275 | 273 | 273 |
| 7 | 2600 | 2597 | 1818 | 390 | 389 |
| 8 | 1419 | 1417 | 992 | 213 | 212 |
| 9 | 1865 | 1863 | 1304 | 280 | 279 |
| **Total** | **17,675** | **17,655** | **12,359** | **2,650** | **2,646** |

The resulting proportions are approximately 70.003%, 15.010%, and 14.987%; small differences are due to integer per-class allocation. Each of the ten classes appears in every partition.

Use `train_indices.json`, `validation_indices.json`, and `test_indices.json` for all ten models. Do not create another random split. Fit feature scaling, PCA, imputation, and scratch-model normalization on training data only (inside training folds for cross-validation). No learned preprocessing statistics have been fitted during preparation. Keep the test set closed until choices are frozen.

Independent validation checked source/manifest hashes, labels against raw data, exclusions, complete/disjoint membership, per-class counts, and absence of known candidate edges crossing partitions. Run `python validate_data_preparation.py` to repeat it. This checks recorded invariants, not astronomical label correctness.

### Files and storage

- `metadata/data_v1/sample_manifest.csv`: every raw row, inclusion flag, reason, label, group, metadata flags, split.
- `metadata/data_v1/review_decisions.json`: visual review provenance, groups and exclusions.
- `metadata/data_v1/duplicate_candidates.csv`: 18 final review pairs.
- `metadata/data_v1/initial_screen_matches.csv` and `image_screen_assessment.csv`: complete screening trail.
- `metadata/data_v1/scan.json`: source identity, method thresholds, versions and cache checksum.
- `metadata/data_v1/summary.json`: frozen seed, counts, hashes, limitations.
- `metadata/data_v1/{train,validation,test}_indices.json`: original-row indices for loading.
- `data/processed/preparation_v1/`: ignored caches and visual review sheets.

Raw images and local caches stay under ignored `data/`. Small text metadata are under `metadata/` for deliberate version control. This preparation was performed locally; Kamiak environment setup and data transfer have not been verified. No repository upload is implied.
