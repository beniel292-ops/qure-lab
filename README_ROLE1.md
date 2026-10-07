# QURE Lab — Role 1 (Quantum) package

A four-qubit Qiskit classifier with documented noise settings, readout-error mitigation, seeded repeats and schema-valid export. **Status: implemented, tested, and run on the real LIDC subset (see `README.md` and `docs/TEAM_STATUS.md`).**

## Run it

**Google Colab (team preference):** open `notebooks/02_quantum_model.ipynb`, then `notebooks/03_noise_and_mitigation.ipynb`. The first cell clones branch `quantum-core` and installs `requirements-quantum.txt`. If the branch isn't pushed yet, it asks you to upload this package as a zip.

**One command** (Colab terminal or any Python 3.10+):

```bash
pip install -r requirements-quantum.txt
python -m pytest -q tests/                                     # 12 tests
python -m src.quantum.run_experiment --fixture --split val     # synthetic smoke test (~25 s)
python -m src.quantum.run_experiment --fixture --split val --no-entangle    # entanglement ablation
# real data (after Role 2 delivers):
python -m src.quantum.run_experiment --features data/features.csv --meta data/features_meta.json --split val
# final, once, with the frozen model:
python -m src.quantum.run_experiment --features ... --meta ... --split test --final --model results/models/<file>.json
```

## Files

| Path | What |
| --- | --- |
| `src/quantum/config.py` | noise settings N0–N4, shots, seeds, mitigation description |
| `src/quantum/data_io.py` | input contract loader, patient-leakage check, synthetic fixture generator |
| `src/quantum/model.py` | circuit, encoding, exact evaluator, circuit metadata |
| `src/quantum/train.py` | training (train split), restart and threshold selection (val split), frozen model files |
| `src/quantum/execute.py` | Aer noise models, calibration, tensored readout mitigation, repeats, state diagnostics |
| `src/quantum/export.py` | experiment records, quantum-side summary, schema validation |
| `src/quantum/run_experiment.py` | the one-command entry point (refuses `test` without `--final` and a frozen model) |
| `schemas/experiment.schema.json` | proposed record format v0.1.0 (Roles 2 and 3 to confirm) |
| `docs/QUANTUM_INPUT_CONTRACT.md` | what Role 2 must deliver |
| `docs/QUANTUM_METHOD.md` | technical explanation, limits, fixture observations, sources |
| `docs/ROLE1_HANDOFFS.md` | copyable messages for Roles 2, 3 and 4 |
| `results/fixtures/` | SYNTHETIC runs for UI development (never evidence) |
| `results/models/` | frozen fixture models (retrain on real data) |

## Suggested branch

Commit this on `quantum-core`, then open a pull request to `main` after Role 2 confirms the input contract and Role 3 confirms the schema.
