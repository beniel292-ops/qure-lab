# Claims checklist (verified / planned / unsupported)

Every number below comes from `results/final/final_summary.json` (test split, run once with frozen settings) or `results/metrics/classical_*.json`. Results are simulated on Qiskit Aer 0.17.2 under **hypothetical** noise settings.

## VERIFIED (implemented, run, and in saved files)

| Claim | Evidence |
| --- | --- |
| A 4-qubit Qiskit circuit (12 RY angles, 6 CX, transpiled depth 26) classifies the LIDC suspicion category | `circuit` block in the summary; tests |
| Noiseless test balanced accuracy 0.857 (95% CI 0.806–0.902); sensitivity 0.883, specificity 0.832, AUC 0.882 | `quantum_test` N0 |
| Logistic regression on the same 4 features: BA 0.852 (0.802–0.901), AUC 0.919 | `classical_test` B1 |
| The quantum model is in the same range as simple classical models; the classical model ranks cases better (higher AUC) | the two rows above |
| Without entangling gates the circuit reaches only 0.607 test BA (with this parity readout) | `ablation_no_entanglement_test` |
| High noise (N4): mean deviation from the noiseless score 0.180 → 0.056 with readout mitigation | `quantum_reliability_test` N4 |
| High noise (N4): decision flips vs noiseless 4.7% → 1.6% | same |
| Mitigation increases run-to-run spread: std 0.0098 → 0.0180; raw with the same total shots 0.0056 | same |
| Gate-only noise (N2): mitigation gives essentially no improvement (0.0203 → 0.0197) | N2 row |
| Mitigation produced 1 run with a score outside [0, 1] at N1; it is kept, not clipped | N1 row |
| Instability rule U, pre-declared: at N4 it flags 4.9% of cases and catches 8 of 12 unstable cases (precision 89%). Flagged cases had a 44% label-error rate vs 14% for retained cases (n = 9 flagged) | `instability_test` |
| Platt-calibrated noiseless probability: test Brier 0.120 | `calibration` |
| Patient-level split with no leakage; the test split was used once | loader check; `EXPERIMENT_PROTOCOL.md` |

## PLANNED (not done)

- Zero-noise extrapolation (stretch)
- Noise settings derived from a real backend's calibration data
- A real IBM hardware run
- An optional Quest/WebXR view of the same data
- Patient-clustered bootstrap intervals

## UNSUPPORTED — never say these

- Quantum advantage, or "the quantum model is better"
- Clinical validity, diagnosis, or cancer detection. The label is radiologist suspicion, not pathology
- "Mitigation fixes the answer" or "error-corrected". It is readout mitigation only, and it can add variance
- Bloch-vector length as confidence or trust
- A single overall "trust score"
- That the website runs live simulations. It replays saved runs

## Added after the Role 2 merge (primary cohort, CIRDataset)

- Supported: on 90 validation cases, the 4-qubit classifier (balanced accuracy 0.857, AUC 0.911) and logistic regression (0.810 at threshold 0.5, or 0.845 val-tuned; AUC 0.897) are **not distinguishable**. The paired difference CI is [−0.022, 0.050].
- Supported (simulation, hypothetical noise): at N4, readout mitigation reduces the mean score deviation from 0.141 to 0.041. It also increases the run-to-run spread, from 0.011 to 0.019.
- Not yet supported: any test-split accuracy on this cohort. Role 2 holds the labels and has not scored the frozen predictions yet.
- Caveat: the quantum restart and threshold were chosen on validation, so validation accuracy is optimistic.
- Caveat: the entanglement ablation result differs between cohorts (0.846 here vs 0.607 on LIDC). Do not generalise either way.
