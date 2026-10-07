# Role 1 handoff messages (copy into each teammate's chat)

> **UPDATE (later the same day):** real data (`data/lidc`, LIDC-IDRI annotations) and a final test run now exist. See `docs/TEAM_STATUS.md`. The messages below are still valid for the contracts; the fixture-only caveats now apply only to `results/fixtures/`.

## → Role 2 (Data and Evaluation)

> **From Role 1 (Quantum):** the quantum pipeline is implemented and tested end to end on a SYNTHETIC fixture. I need your real input files in the format in `docs/QUANTUM_INPUT_CONTRACT.md`:
> - `data/features.csv`: case_id, patient_group_id, split, label, plus 4 feature columns standardized on train only
> - `data/features_meta.json`: source, version, label definition, positive-label meaning, feature names, preprocessing note
>
> My loader rejects patient leakage, non-finite values and a missing train or val split.
>
> Three questions for you:
> 1. Does your source have real patient IDs? NoduleMNIST3D does not.
> 2. Are the 4 features radiologist-rated characteristics (correlated with the suspicion label)? If so, please document that.
> 3. What are the counts and class balance per split?
>
> What I give back:
> - per-case raw quantum scores (noiseless / noisy / mitigated, 5 seeded repeats), flip rates and deviations, in `results/runs/*.json` (schema `schemas/experiment.schema.json`)
> - **empty** `calibrated_probability` and `classical_baseline` fields for you to fill when you merge, on the same case_ids
>
> `raw_score` is NOT a calibrated probability. If you calibrate, fit on validation only and name the method. Please define any "unstable under tested conditions" rule on the validation runs; I only touch test with `--final` and a frozen model.

## → Role 3 (Website and VR)

> **From Role 1 (Quantum):** build against `results/fixtures/FIXTURE-qml4-L2-ent-val-*.json`. It is SYNTHETIC: `data_status: "synthetic-fixture"`, and the run has a top-level `warning`. Show a visible banner whenever `data_status != "real"`.
>
> Structure: one run file contains `records[]`, one record per case × noise setting (N0–N4). Schema: `schemas/experiment.schema.json`.
>
> Per record, show:
> - `features.names/values`, `encoding.angles_rad`, `circuit.text_drawing` (top-level `circuit`), `noise.params` and `noise.source` ("hypothetical simulator sweep")
> - `outputs.noiseless_exact.raw_score`, `outputs.noisy.values` (5 repeats), `outputs.mitigated.values` (5 repeats, may fall outside [0, 1]; show them as-is), `outputs.noisy_equal_budget`
> - `decision_threshold`, `flip_rate_*`, `reference_label`, `noiseless_correct_vs_label`, `execution.budget`
>
> Rules:
> - The noise slider = **precomputed playback** of N0–N4. Label it that way, and never interpolate between settings.
> - Call the third result "mitigated estimate", never "fixed".
> - `state_diagnostics` (Bloch vectors) must sit in a separate "state diagnostics" panel with its note. They are not confidence. There is no mitigated Bloch vector, by design.
> - Show unfavourable cases too. Mitigation sometimes moves scores farther away, and in the fixture it raised flip rates under readout noise.
> - `classical_baseline` and `calibrated_probability` are `null` until Role 2 merges them. Show "pending", not a number.

## → Role 4 (Pitch and Documentation)

> **From Role 1 (Quantum)** — claim status:
>
> **VERIFIED (implemented and tested on a synthetic fixture; real data pending):**
> - 4-qubit Qiskit classifier
> - entanglement ablation
> - 5 documented noise settings (hypothetical, not hardware-derived)
> - readout mitigation with calibration
> - 5 seeded repeats
> - equal-shot-budget comparison
> - schema-valid export
> - 12 automated tests
>
> **SAFE TO SAY:** "Mitigation targets readout error only; it does not remove gate noise, and it can add variance."
>
> **DO NOT SAY YET:** any accuracy or improvement number. The fixture numbers in `docs/QUANTUM_METHOD.md` §6 are synthetic. Use "results pending".
>
> **A good slide idea, if the real data confirms it:** "Closer on average is not the same as more stable decisions." In the fixture, mitigation removed readout bias but increased decision flips.
>
> Method and sources are in `docs/QUANTUM_METHOD.md`. Please have me review technical slides.
