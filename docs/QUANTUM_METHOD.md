# Quantum method (Role 1) — what is implemented and tested

Status labels follow the team rule: PLANNED ≠ IMPLEMENTED ≠ TESTED ≠ CLINICALLY VALIDATED. Everything here is **IMPLEMENTED and TESTED**, and it has been run on the real LIDC-IDRI subset (`data/lidc`). Real-data results are in `results/final/` and `docs/CLAIMS_AND_LIMITATIONS.md`. Section 6 below covers the synthetic development fixture only.

## 1. Classifier (hybrid: quantum circuit + classical optimizer)

| Part | Choice | Why |
| --- | --- | --- |
| Input | 4 standardized features from Role 2 | 4 qubits; the classical preprocessing is shown explicitly, never hidden |
| Encoding | RY(2·arctan xᵢ) on qubit i | bounded, smooth, monotonic; no clipping of outliers |
| Ansatz | 2 layers of [RY(θ) on each qubit → CX 0-1, 1-2, 2-3], then a final RY layer | shallow: 12 parameters, 6 CX; transpiled depth is in every record |
| Readout | measure all 4 qubits; `raw_score` = P(odd parity) = (1 − ⟨ZZZZ⟩)/2 | uses every qubit, so the no-entanglement ablation stays meaningful |
| Training | class-balanced cross-entropy on exact noiseless outputs; SciPy COBYLA; 3 seeded restarts | parameters are frozen afterwards so execution noise can be isolated |
| Selection | best restart by **validation** loss; threshold by **validation** balanced accuracy | the test split is never used for any choice |
| Ablation | the same circuit without CX gates | we test, not assume, whether entanglement helps |

`raw_score` is a measurement-derived number. **It is not a calibrated probability.** If Role 2 fits a calibrator (Platt or isotonic) on validation data, it goes in the separate `calibrated_probability` field, together with its method.

Correctness checks:

- The fast exact training path (Qiskit `Operator` for the ansatz × product encoding states) equals the full-circuit Qiskit `Statevector` result to 3e-16 (test plus notebook cell).
- With zero noise, the infinite-shot noisy path equals the noiseless output to 1e-10.
- The same seeds give identical counts.

## 2. Noise settings (hypothetical simulator sweep)

| ID | 1-qubit depolarizing (sx, x) | CX depolarizing | P(read 1 given 0) | P(read 0 given 1) |
| --- | --- | --- | --- | --- |
| N0 | 0 | 0 | 0 | 0 |
| N1 | 0 | 0 | 0.02 | 0.04 |
| N2 | 0.001 | 0.01 | 0 | 0 |
| N3 | 0.001 | 0.01 | 0.02 | 0.04 |
| N4 | 0.003 | 0.03 | 0.05 | 0.08 |

These values are **not** derived from hardware calibration data. They are a controlled sweep that separates readout noise (N1) from gate noise (N2) and combines both (N3, N4). `rz` is treated as noiseless (virtual). A setting derived from a real backend's calibration snapshot would be a separate, labelled experiment.

## 3. Execution outputs per case and setting

- `noiseless_exact`: the reference. It can still be wrong against the dataset label, and records show `noiseless_correct_vs_label`.
- `noisy_exact`: the infinite-shot noisy expectation (density matrix plus analytic readout). It shows the effect of the noise, separated from shot noise.
- `noisy`: 5 seeded repeats × 2000 shots.
- `noisy_equal_budget`: 5 repeats × 6000 shots, the same total budget as mitigation (2000 + 2 × 2000 calibration). Counting the shared calibration per case is conservative, in favour of the raw estimate.
- `mitigated`: tensored readout mitigation. Per-qubit 2×2 matrices are calibrated from |0000⟩ and |1111⟩ (2000 shots each, recalibrated every repeat), inverted, and applied to the measured distribution.
- **Not clipped:** negative quasi-probabilities and scores outside [0, 1] are kept and counted (`negative_quasiprob_repeats`, `score_out_of_range_repeats`).
- Decisions use the frozen threshold. `flip_rate_*` is the fraction of repeats whose decision differs from the noiseless decision.

## 4. What readout mitigation can and cannot do

- **It targets** measurement errors only, assuming they are independent per qubit (true in this simulator).
- **It does not** undo gate or decoherence errors. Under N2 (gate only) it should change almost nothing.
- **It costs** extra calibration shots, and it typically **increases** repeat-to-repeat variance, because the inverse matrix amplifies sampling noise.
- **Caveat:** the X gates that prepare |1111⟩ are noisy under gate-noise settings, so preparation error leaks into the calibration.
- **It can** restore the noiseless answer even when that answer is wrong against the label.

## 5. State diagnostics (separate from reliability)

Per-qubit Bloch vectors come from the exact noiseless state and the gate-noisy density matrix. Entanglement alone shortens them, and different noise channels act differently. They are **state diagnostics, not confidence or trust measures**. Readout mitigation does not change the state, so there is no "mitigated Bloch vector".

## 6. Development-fixture observations (synthetic, NOT evidence)

These are recorded only to show what the pipeline can surface. Real numbers will differ.

Numbers come from the full fixture run (38 validation cases, 5 repeats, 2000 shots).

| Fixture observation | Value |
| --- | --- |
| Decisions flipped by finite shots alone (N0, zero noise) | 2.1% of repeats (cases near the threshold) |
| Readout-only (N1): mean deviation from noiseless | 0.067 noisy → 0.006 mitigated; closer in 95% of cases |
| Readout-only (N1): decision flip rate | **0.5% noisy vs 2.6% mitigated**: mitigation removed the bias but added variance, which flipped more near-threshold decisions |
| Gate-only (N2): mean deviation | 0.0235 noisy vs 0.0228 mitigated: essentially no help, as expected |
| High noise (N4): mean deviation | 0.169 noisy → 0.065 mitigated |
| High noise (N4): repeat std | 0.011 noisy, 0.020 mitigated, 0.007 raw at equal shot budget |
| Negative quasi-probabilities | present in many mitigated repeats; no score left [0, 1] in this run |
| No-entanglement ablation, validation balanced accuracy | 0.58 vs 0.87 entangled |

The N1 flip-rate result is exactly the kind of finding the project exists to show: **closer on average is not the same as more stable decisions.**

## 7. Sources

- Temme, Bravyi, Gambetta, *Error mitigation for short-depth quantum circuits* (2017): https://arxiv.org/abs/1612.02058
- IBM, error mitigation and suppression techniques: https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques
- IBM, suppression vs mitigation vs correction: https://www.ibm.com/quantum/blog/quantum-error-suppression-mitigation-correction
- IBM, plotting quantum states: https://quantum.cloud.ibm.com/docs/en/guides/plot-quantum-states
- scikit-learn, probability calibration: https://scikit-learn.org/stable/modules/calibration.html
