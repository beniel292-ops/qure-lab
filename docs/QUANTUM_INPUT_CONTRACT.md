# Quantum input contract v0.1.0 (Role 2 → Role 1)

Status: **PROPOSED by Role 1.** Role 2 should confirm it or reply with changes. The quantum code validates every rule below when it loads the files and refuses bad input.

## File 1: `data/features.csv`

One row per **case** (nodule-level unit). Columns:

| Column | Type | Rule |
| --- | --- | --- |
| `case_id` | string | unique, de-identified |
| `patient_group_id` | string | de-identified patient ID; **all cases of one patient in one split** (checked; leakage raises an error) |
| `split` | `train` / `val` / `test` | patient-level split, fixed seed, recorded by Role 2 |
| `label` | 0 / 1 | the dataset's reference label as defined in the meta file |
| 4 feature columns | float | names listed in `feature_names`; **standardized with train-split statistics only**; finite values |

## File 2: `data/features_meta.json`

```json
{
  "input_contract_version": "0.1.0",
  "data_status": "real",
  "dataset_source": "e.g. LIDC-IDRI annotations via <tool/version>, URL",
  "dataset_version": "e.g. download date / DOI / commit",
  "label_definition": "e.g. mean radiologist malignancy-suspicion rating >= 4 vs <= 2; rating 3 excluded",
  "label_positive_meaning": "e.g. 'high suspicion category in this dataset' (NOT pathology-confirmed unless it is)",
  "feature_names": ["feat1", "feat2", "feat3", "feat4"],
  "preprocessing_version": "e.g. prep-v1",
  "preprocessing_note": "how the 4 features were derived, scaler fitted on train only, any exclusions"
}
```

`data_status` must be `"real"` for anything shown to judges. The synthetic fixture uses `"synthetic-fixture"`.

## What Role 1 does with it

- Encoding: angle = 2·arctan(x), applied per feature. Standardized values map smoothly into (−π, π), with no clipping.
- **Train** fits the parameters. **Validation** picks the restart and the decision threshold, and is used for exploring results. **Test** is used only by the final, flag-guarded run with a frozen model.
- Role 1 never reads labels from any other source and never uses label-derived fields as inputs.

## Questions for Role 2 (they affect the quantum side)

1. **Patient IDs.** Does the chosen source really have patient identifiers? Some prepackaged nodule datasets (for example MedMNIST's NoduleMNIST3D) do not, so a patient-level split cannot be verified with them.
2. **Feature provenance.** If the 4 features are radiologist-rated characteristics (spiculation, margin, …), they come from the same readers who gave the suspicion rating. That is not label-derived, but it is strongly correlated human judgement. Please state it in the dataset card.
3. **Class balance and counts** per split. The quantum loss is class-balanced, but very small splits (fewer than about 30 cases per class in val) make threshold choice and flip-rate estimates noisy.
