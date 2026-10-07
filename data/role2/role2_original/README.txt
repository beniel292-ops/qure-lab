QURE ROLE 2 HANDOFF — 2026-10-06
597 patients: train 417, validation 90, test 90. No patient overlap.
Source: CIRDataset record 6762573, prepared NumPy CT/mask archive.
Target: radiologist suspicion (low ratings 1–2 vs high ratings 4–5), not cancer diagnosis.
Role 1: consume theta_0..theta_3 IN ORDER as angles in radians. Do not rescale again.
Use target only from train/validation files. Test input labels are intentionally withheld.
Compare your validation outputs against validation_classical_predictions.csv by case_id.
Classical baseline uses the exact same four angle inputs; its scores are not clinically calibrated.
Role 3: linked_validation_case.png and linked_case.json refer to a genuine validation case.
Preprocessing parameters and feature definitions are in preprocessing.json.
Validation balanced accuracy: 0.8103; AUC: 0.8967. Preliminary, not final test results.
No quantum advantage or mitigation result is claimed.
Test labels stay with Role 2 until models and thresholds are frozen.
Raw cloud files disappear when runtime resets; rerun the notebook to regenerate.
Cite https://zenodo.org/records/6762573 and https://www.cancerimagingarchive.net/collection/lidc-idri/.
