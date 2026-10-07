# Pitch (first round, about 2 minutes) — numbers checked against `results/final/final_summary.json`

**Opening (problem).** A quantum model can give a convincing answer even after noise has changed it. Before anyone trusts quantum classifiers on hardware, we need to see when that happens and what fixing it costs.

**What QURE Lab is.** A reproducible research environment that runs a four-qubit Qiskit classifier on real lung-nodule annotation data from LIDC-IDRI. It shows the noiseless, noisy and readout-mitigated results side by side, next to a classical baseline.

**Data constraints we handled.**

- 879 nodules from 535 patients, split by patient so no patient appears in two splits.
- The label is the radiologists' suspicion rating, not pathology, and we say so.
- Four geometric features from the reader outlines feed four qubits.
- The test split was run once, with frozen settings.

**Results (simulated, hypothetical noise).**

- **Accuracy:** the quantum model matches a logistic regression on the same features (balanced accuracy 0.857 vs 0.852). We claim no advantage.
- **Noise:** under our highest noise setting, 4.7% of decisions flip.
- **Mitigation:** readout mitigation cuts the deviation from the noiseless answer by about three times, and flips to 1.6%.
- **Cost:** mitigation uses 4,000 extra calibration shots and roughly doubles run-to-run spread. It does nothing for gate noise.
- **Detecting instability:** a simple pre-declared instability flag catches two thirds of the unstable cases using only information a hardware user would have.

**Feasibility.** Four qubits, Colab-runnable, one command to reproduce, 12 automated tests, and a desktop dashboard. VR is optional.

**Close.** Our contribution is not a diagnostic tool. It is a way to measure when a quantum model's answer can be relied on, and what that reliability costs.

## Five-minute structure

| Time | Segment |
| --- | --- |
| 0:00–0:30 | Problem |
| 0:30–1:15 | Data and pipeline (dashboard: Method tab) |
| 1:15–3:00 | Live demo (dashboard: Case explorer on a flagged case, switching N0 → N4) |
| 3:00–4:00 | Aggregate results and cost |
| 4:00–4:30 | Limitations |
| 4:30–5:00 | What's next: ZNE, real-device noise, hardware run |

## Likely judge questions

- **"Is this really quantum?"** Yes. The classifier's output is measured from a Qiskit circuit. Training and data handling are classical, which is normal for hybrid models.
- **"Why not just use logistic regression?"** For accuracy, you could. The question we study is reliability under quantum noise, which classical models don't have.
- **"Is the noise realistic?"** It is a documented hypothetical sweep that separates readout and gate noise. Calibration-derived noise is our next step.
- **"Does mitigation fix the answer?"** Partly. It removes readout bias, adds variance, and doesn't touch gate noise.
- **"Can you diagnose cancer?"** No. The label is radiologist suspicion. This is a research tool.
