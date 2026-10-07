# Role 2 merge — CIRDataset cohort (2026-10-06)

Role 2 delivered a reproducible notebook with an embedded handoff ZIP. It is now the **primary cohort**. The Role 1 LIDC reader-contour cohort stays as a clearly labelled **replication cohort**.

## What Role 2 delivered

- 597 LIDC-IDRI patients, one canonical CIRDataset patch each (CT patch plus expert nodule mask; Zenodo 6762573, CC BY 4.0).
- Label: radiologist suspicion, ratings 1–2 → 0 and 4–5 → 1. Rating 3 is excluded. This is not pathology-confirmed cancer.
- Four image/mask features: `log_mask_voxels`, `mean_stored_intensity`, `std_stored_intensity`, `bbox_fill_fraction`.
- Angles θ = π·sigmoid((x − train_mean)/train_scale), using train-only statistics.
- Patient-level split 417 / 90 / 90. **Test labels are withheld** until the models are frozen.
- Logistic regression on validation: balanced accuracy 0.810, AUC 0.897.

## What Role 1 changed (nothing of Role 2's was edited)

| File | Change |
| --- | --- |
| `src/data/role2_handoff.py` | New. Reshapes the handoff into `data/role2/features.csv` + `features_meta.json`. It cross-checks the split manifest and refuses any test label. Originals are copied to `data/role2/role2_original/`. |
| `src/quantum/model.py` | `encode_angles(x, encoding)`: `arctan` (old) or `precomputed_angles` (Role 2; used as-is). |
| `train.py`, `execute.py`, `export.py`, `run_experiment.py` | The encoding is threaded through and stored in the frozen model. The CLI refuses a model whose encoding does not match the data. |
| `data_io.py` | Reads `input_encoding`. Checks that angles lie in [0, π]. A blank label is allowed only on test (stored as −1, exported as `null`). |
| `schemas/experiment.schema.json` | `reference_label` may be `null`. |
| `src/evaluation/role2_report.py` | New. Validation comparison and label-free test reliability. Writes the frozen test predictions. |
| `tests/test_quantum.py` | Two new tests: the precomputed-angle path matches the Qiskit statevector, and the Role 2 data contract holds (test labels withheld). **15/15 pass.** |
| `web/template.html`, `tools/build_web_bundle.py` | Website rebuilt as a dark console UI. The old LIDC page is kept as `web/template_lidc_v1.html` + `tools/build_web_bundle_lidc.py`. |

## Results (simulated; validation n = 90)

The quantum model's restart and threshold were selected on validation, so its validation balanced accuracy is optimistic. The row "Logistic regression, val-tuned" carries the same optimism.

| Model | Threshold | Bal. acc. (95% CI) | AUC |
| --- | --- | --- | --- |
| 4-qubit VQC, CX chain, noiseless | 0.39 (val) | 0.857 (0.780–0.928) | 0.911 |
| 4-qubit VQC, no CX (ablation) | 0.49 (val) | 0.846 (0.774–0.913) | 0.893 |
| Logistic regression (Role 2) | 0.5 | 0.810 (0.730–0.890) | 0.897 |
| Logistic regression, val-tuned | 0.34 (val) | 0.845 (0.772–0.916) | 0.897 |

The paired bootstrap difference (quantum − val-tuned logistic regression) is +0.011, 95% CI [−0.022, 0.050]. **The models are not distinguishable** on this cohort, and no advantage is claimed. Unlike the LIDC cohort (0.607), removing the CX gates barely matters here.

Noise and mitigation results at N4 (gate + high readout noise):

| Split | Mean \|Δ\| noisy → mitigated | Flip rate noisy → mitigated | Std noisy → mitigated (equal-budget raw) |
| --- | --- | --- | --- |
| Validation | 0.141 → 0.041 | 2.7% → 0.9% | 0.011 → 0.019 (0.006) |
| Test | 0.131 → 0.038 | 16.9% → 3.3% | 0.010 → 0.020 (0.006) |

Instability rule U on test, which is label-free:

- At N4, U flags 16.7% of cases with precision 100%, and recalls 83% of cases that flipped in simulation.
- On validation, flag counts are small (1–3 cases per setting), so no strong claim is possible.

## Test handoff → Role 2

- **File:** `results/role2/final/test_predictions_for_role2.csv`. It has 90 rows: the noiseless score/decision, the Platt probability (fit on val), N0–N4 noisy/mitigated means, mitigated decisions, U flags, and ablation scores.
- **Frozen models:**
  - `results/role2/models/qml4-L2-ent-20261006-164727.json` (threshold 0.39)
  - `results/role2/models/qml4-L2-noent-20261006-164718.json` (threshold 0.49)

> Role 2: models and thresholds are frozen. Please score `noiseless_decision` as the primary prediction. Also score each `N*_mitigated_decision`.
> Report balanced accuracy, sensitivity, specificity, AUC on `noiseless_raw_score`, and Brier on `platt_probability`.
> Score your logistic regression on the same 90 test cases. Send back the metrics JSON only, not the labels.

## Reproduce

```bash
python -m src.data.role2_handoff --src <extracted handoff> --out data/role2
F="--features data/role2/features.csv --meta data/role2/features_meta.json --out results/role2"
python -m src.quantum.run_experiment $F --split val
python -m src.quantum.run_experiment $F --split val --no-entangle
python -m src.quantum.run_experiment $F --split test --final --model results/role2/models/<ent>.json
python -m src.quantum.run_experiment $F --split test --final --model results/role2/models/<noent>.json
python -m src.evaluation.role2_report --val-run ... --test-run ... --ablation-val-run ... --ablation-test-run ...
python tools/build_web_bundle.py
```
