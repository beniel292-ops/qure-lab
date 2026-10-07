# Experiment protocol (frozen before the test split was run)

**Frozen on 2026-10-06, before any test-split execution.** Everything below was decided using the train and validation splits only.

1. **Data:** `data/lidc` built by `src/data/build_lidc_dataset.py` (see `docs/DATASET_CARD.md`). Patient-level split 60/20/20, seed 2026.
2. **Quantum model:** `qml4-L2-ent` (2 layers, CX chain, parity readout). Parameters were trained on train, the restart chosen by validation loss, and the threshold by validation balanced accuracy (0.40). The model file in `results/models/` is frozen.
3. **Ablation:** `qml4-L2-noent`, the same protocol with no CX gates. It is reported, not selected.
4. **Noise settings:** N0–N4 as in `src/quantum/config.py` (a hypothetical simulator sweep). Each case and setting gets 2000 shots × 5 seeded repeats. Mitigation recalibrates every repeat with 2 × 2000 shots, and the equal-budget raw run uses 6000 shots.
5. **Classical baselines (fixed hyperparameters, no search):**
    - B1 logistic regression on the same 4 features
    - B2 RBF-SVM on the same 4 features
    - B3 gradient boosting on 7 geometric features (stronger reference)
    - R1 logistic regression on reader semantic ratings (reference only, unfair)

   Thresholds come from validation.
6. **Instability rule U (pre-declared, not tuned):** flag a case at a setting if either
    - the raw-noisy and mitigated repeat means give different decisions, or
    - |mitigated mean − threshold| < 2 × the standard error of the mitigated mean.

   U uses only quantities observable on hardware. It is evaluated against the simulation-only ground truth "any noisy repeat decision differs from the noiseless decision".
7. **Calibration:** Platt scaling on validation noiseless raw scores. It is reported on test (Brier score and a reliability table) and applies to noiseless scores only.
8. **Test:** run once with `--final` and the frozen model. No changes after viewing test results; any later change must be reported as a new, separately labelled experiment.
9. **Reported uncertainty:**
    - 95% bootstrap CIs for balanced accuracy (case-level; nodules from the same patient are correlated, so the CIs are optimistic)
    - repeat standard deviations for the finite-shot runs
