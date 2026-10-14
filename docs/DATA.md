# Dataset plan

**Status: access documentation only.** The dataset has not been downloaded or inspected by this repository.

## Source

- Dataset: Galaxy10 DECaLS, compiled by Henry Leung and Jo Bovy.
- Planned file: `Galaxy10_DECals_NoDuplicated.h5`.
- Download record: https://zenodo.org/records/10845026
- Dataset documentation: https://astronn.readthedocs.io/en/latest/galaxy10.html
- Dataset DOI cited in the proposal: https://doi.org/10.5281/zenodo.10845025
- Images: DESI Legacy Imaging Surveys; labels: Galaxy Zoo volunteer classifications.

The proposal identifies the dataset license as Creative Commons Attribution 4.0 (CC BY 4.0). Verify the selected file's record and license before reuse, and cite the dataset creators and record. This dataset license does not establish a software license for the repository.

## Expected fields

These expectations come from the proposal and require verification against the downloaded file.

| Field | Planned use |
| --- | --- |
| `images` | 256 x 256 color image input; g, r, and z band composites |
| `ans` | Class label, 0 through 9 |
| `ra`, `dec` | Sky coordinates for duplicate checks |
| `redshift` | Error analysis and a separate metadata ablation |
| `pxscale` | Zoom-level analysis and a separate metadata ablation |

The original release contains 17,736 images. The deduplicated count and per-class counts are unknown until inspected. Never substitute the original counts for the cleaned dataset.

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

## Planned access and verification

1. Download the deduplicated file to the research compute environment.
2. Record the source URL, retrieval date, filename, checksum algorithm, and expected checksum from the source.
3. Verify file integrity, field names, array lengths, image shape, label range, and missing metadata.
4. Check sky-position proximity and perceptual image hashes; inspect flagged matches before removing true duplicates.
5. Save the cleaning decisions and one stratified 70/15/15 split with a fixed seed. Any confirmed duplicate group must stay within one partition.
6. Record exact retained sample counts and class counts.

Store raw and processed data locally under `data/`, which is ignored by Git. Small reproducibility records such as split indices and checksum metadata can later be stored deliberately under `metadata/`, after review. Do not upload the dataset or large derived feature arrays to GitHub.
