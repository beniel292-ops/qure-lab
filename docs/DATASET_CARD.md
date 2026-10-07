# Dataset card — QURE Lab LIDC-IDRI nodule subset (`data/lidc`, preprocessing `lidc-geom-v1`)

**Status (updated):** this LIDC reader-contour cohort is now the **replication cohort**. The primary cohort is Role 2's CIRDataset handoff (597 patients, CT patches with expert masks, test labels held by Role 2). Its card is in `data/role2/role2_original/DATASET_CARD.json` and is summarised in `docs/ROLE2_MERGE.md`.

Originally built by Role 1 as a draft for Role 2 to verify or replace. It is reproducible with `python -m src.data.build_lidc_dataset --out data/lidc` (about 4 minutes, no CT download).

| Item | Value |
| --- | --- |
| Source | LIDC-IDRI radiologist annotations (XML-derived), from the database bundled with `pylidc` 0.2.3 |
| Collection | https://www.cancerimagingarchive.net/collection/lidc-idri/ (licence CC BY 3.0; cite the LIDC-IDRI papers per TCIA's citation policy) |
| Image data used | **None.** No CT pixels were downloaded; the full archive is about 133 GB |
| Unit | one physical nodule = a pylidc cluster of annotations by different readers on one scan |
| Inclusion | nodules annotated (contoured) by ≥ 3 readers |
| Label | median of the readers' malignancy ratings (1–5): > 3 → 1, < 3 → 0 |
| Label meaning | radiologist **suspicion** category, not pathology or biopsy |
| Exclusions | 1,259 clusters with < 3 readers; 513 with median rating = 3 (indeterminate); 0 geometry errors |
| Final size | 879 nodules from 535 patients |
| Split | patient-level random 60/20/20, seed 2026 |
| Train | 505 nodules / 321 patients (254 high, 251 low) |
| Validation | 191 / 107 (82 high, 109 low) |
| Test | 183 / 107 (94 high, 89 low) |
| Quantum inputs (4) | mean over readers of: log volume (mm³), compactness 36πV²/A³, log elongation (contour principal axes), flatness (z-extent / in-plane extent) |
| Scaling | z-score with **train** statistics only (stored in `features_meta.json`) |
| Feature choice | pre-specified before looking at labels; geometric only |
| Other columns (`nodules_full.csv`) | 7 geometric features (B3 baseline) and 8 median semantic ratings (R1 reference only); never used as quantum inputs |
| De-identification | IDs are the public LIDC patient IDs (already de-identified by TCIA) |

## Known limitations

- **Contour-derived features:** the features come from reader-drawn contours, so they are radiologist-derived geometry. The label comes from the same readers' ratings. That is not leakage of the label field, but the two are correlated human judgements; readers tend to rate larger nodules as more suspicious.
- **Small nodules:** nodules under 3 mm (no contours) and those seen by fewer than 3 readers are excluded, so the sample is not the full population.
- **Indeterminate cases removed:** excluding median rating 3 removes the hardest cases. This makes the task easier than real triage.
- **Correlated nodules:** several nodules can come from one patient. Case-level bootstrap intervals are therefore somewhat optimistic.
- **No linked CT image:** the website shows reader contour points only, labelled as such.
