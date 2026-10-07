# QURE Lab — Quantum Uncertainty-aware Research Environment

**Can we trust a noisy quantum classifier?** QURE Lab runs a 4-qubit Qiskit classifier on real lung-nodule scans from 597 patients. It adds simulated hardware noise and shows when that noise changes the answer. It also shows what readout-error mitigation fixes, and what it costs.

Qiskit Fall Fest 2026 · Track 5 (Open Innovation) · VIT Chennai

> Research and education prototype. Simulated quantum execution under a hypothetical noise sweep. **Not a medical device, not for diagnosis.** No quantum advantage is claimed.

---

## Contents

1. [What it does](#what-it-does)
2. [Results](#results)
3. [Quick start](#quick-start)
4. [The console](#the-console)
5. [How it works](#how-it-works)
6. [Ask QURE (AI assistant) and API keys](#ask-qure-ai-assistant-and-api-keys)
7. [Rebuild everything](#rebuild-everything)
8. [Deploy online](#deploy-online)
9. [Project structure](#project-structure)
10. [Tests](#tests)
11. [Data, honesty and limits](#data-honesty-and-limits)
12. [Team and credits](#team-and-credits)

---

## What it does

| Input | Processing | Output |
| --- | --- | --- |
| CT patches with expert nodule masks (CIRDataset, LIDC-IDRI), 597 patients, split by patient into 417 / 90 / 90 | 4 image features → 4 angles → 4-qubit circuit → 5 noise settings × 5 runs × 2000 shots → raw vs readout-mitigated scores → instability rule U | An interactive console, a research report PDF, per-experiment PDFs, frozen test predictions, and an AI help assistant |

Every case gets **three results**:

- the **noiseless reference** (exact);
- the **noisy execution** (5 repeats);
- the **mitigated estimate** (5 repeats, readout correction only).

---

## Results

Validation set, 90 nodules. Test labels are held by the data owner (Role 2) until scoring.

| Model | Balanced accuracy (95% CI) | AUC |
| --- | --- | --- |
| 4-qubit quantum classifier, noiseless | 0.857 (0.780–0.928) | 0.911 |
| Logistic regression, same inputs, threshold 0.5 | 0.810 (0.730–0.890) | 0.897 |
| Logistic regression, threshold tuned on validation | 0.845 (0.772–0.916) | 0.897 |
| Quantum, no entangling gates (ablation) | 0.846 | 0.893 |

- **Quantum vs classical:** the difference is +0.011, with a paired 95% CI of −0.022 to +0.050. The two are **not distinguishable**.
- **Noise:** at the strongest setting (N4), 16.9% of runs on the test set change the decision.
- **Mitigation:** it cuts the score error from 0.141 to 0.041 and the decision flips from 16.9% to 3.3%. The price is 3× the shots and a larger run-to-run spread (0.011 → 0.019).
- **Rule U:** at N4 it flags 15 of 90 test nodules for review, and all 15 really flipped.

The quantum model's threshold was picked on the same validation cases, so its validation number is slightly optimistic. The fair number is the test score.

---

## Quick start

Tested on macOS with Python 3.13 and Node 22.

### 1. Install (once)

```bash
cd ~/Desktop/IBM          # or wherever you cloned the repo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-quantum.txt
```

### 2a. Open the console with no setup

```bash
open web/qure_lab_standalone.html
```

This works offline. Ask QURE answers from its built-in guide, and the Accuracy-lab PDF export works.

### 2b. Run the full site with the AI assistant (needs Node 18+)

```bash
cd site
npm test                 # 6 assistant tests, no network
node dev_server.js       # http://localhost:8787
```

Keys are read from a private `.env` file in the project folder (see below). Open http://localhost:8787/api/ask to check that both providers show `true`.

### 3. Run the tests

```bash
python -m pytest -q tests/     # 15 passed
```

---

## The console

| Page | What you do there |
| --- | --- |
| **Start** | Three big tasks, the key findings in plain words, common questions, and the report download |
| **Cases** | Pick a nodule and a noise setting (N0–N4) to see its noiseless, noisy and mitigated scores run by run. LIDC-IDRI-0009 shows its CT slice |
| **Noise & mitigation** | How far scores drift, what mitigation fixes, what it costs, and rule U flags. Switch between validation and test (no labels) |
| **Accuracy lab** | Change the threshold, the model, or noiseless/noisy/mitigated, and watch the confusion matrix and formula update. **Download a PDF** of the result plus every change you made |
| **Ask QURE** | The assistant button, bottom right. It answers questions about the data, accuracy, noise and the dashboard |
| **Help & FAQ** | About 25 searchable answers and a glossary. Dotted terms show definitions on hover |
| **Method & limits** | What is built, tested, planned and not claimed |
| **Report** (top bar) | Downloads the 5-page research report PDF |
| **Simple / Detailed** (top bar) | Detailed view adds Models, All results, Data and Test handoff |

---

## How it works

1. **Data (Role 2).**
    - Four image and mask features per nodule: `log_mask_voxels`, `mean_stored_intensity`, `std_stored_intensity`, `bbox_fill_fraction`.
    - Standardised using train-set statistics, then mapped to angles θ = π·sigmoid(z) ∈ (0, π).
2. **Circuit.**
    - RY(θᵢ) encodes each feature on one of 4 qubits.
    - Then 2 layers of trainable RY rotations, each followed by a CX chain (0–1, 1–2, 2–3), then a final RY layer: 12 weights.
    - All 4 qubits are measured. The score is the share of shots with odd parity.
    - Decision: high suspicion if score ≥ 0.39.
3. **Training.**
    - Class-balanced cross-entropy on noiseless train scores, COBYLA optimiser, 3 restarts.
    - The restart and threshold are chosen on validation, then the weights are frozen.
4. **Noise.**
    - Qiskit Aer, basis gates rz/sx/x/cx, no transpiler optimisation.
    - Five hypothetical settings, N0 (none) to N4 (gate + high readout noise).
    - 5 seeded repeats × 2000 shots per case and setting.
5. **Mitigation.**
    - Each qubit gets a 2×2 readout calibration matrix (4000 calibration shots), which is inverted.
    - Fixes readout error only, not gate noise.
    - Values outside [0, 1] are kept, never clipped.
6. **Rule U.**
    - Flags a case if the raw and mitigated decisions differ, or if the mitigated mean is within 2 standard errors of the threshold.
    - Uses only quantities real hardware would give.
7. **Evaluation.**
    - Balanced accuracy, sensitivity, specificity and AUC, with bootstrap 95% CIs.
    - Paired bootstrap against logistic regression.
    - The no-CX ablation.
    - Platt calibration fitted on validation.
8. **One data file feeds everything.** `web/qure_bundle.json` drives the console, the research PDF, the assistant's facts and the VR spec, so every screen shows the same numbers.

More detail: `docs/QUANTUM_METHOD.md`, `docs/ROLE2_MERGE.md`, `docs/EXPERIMENT_PROTOCOL.md`.

---

## Ask QURE (AI assistant) and API keys

The assistant answers in this order:

1. **Built-in guide.** About 25 curated answers generated from the results. Instant, and works offline.
2. **Gemini**, then **Grok** if Gemini fails, through the server function `site/api/ask.js`. Keys stay on the server and never reach the page.
3. **Claude**, inside the published claude.ai page, using the viewer's own account.
4. If no AI is reachable, it falls back to the closest guide answers.

The AI sees only a fact sheet built from the results. It is told not to invent numbers, not to give medical advice, and never to reveal test labels.

**Keys go in a private file, never in code, chat or git.** Create `.env` in the project folder:

```
GEMINI_API_KEY=your-gemini-key
XAI_API_KEY=your-grok-key
# optional
GEMINI_MODEL=gemini-3.6-flash
XAI_MODEL=grok-4.7
QURE_PROVIDER_ORDER=gemini,grok
```

`.env` is listed in `.gitignore`. Model names change often; if a provider returns HTTP 404, update the model name.

---

## Rebuild everything

Re-run the quantum experiment on the Role 2 data. This is optional and takes a few minutes:

```bash
F="--features data/role2/features.csv --meta data/role2/features_meta.json --out results/role2"
python -m src.quantum.run_experiment $F --split val
python -m src.quantum.run_experiment $F --split val --no-entangle
python -m src.quantum.run_experiment $F --split test --final --model results/role2/models/qml4-L2-ent-20261006-164727.json
python -m src.quantum.run_experiment $F --split test --final --model results/role2/models/qml4-L2-noent-20261006-164718.json
R=results/role2/runs
python -m src.evaluation.role2_report \
  --val-run $(ls $R/qml4-L2-ent-val-*.json | tail -1) --test-run $(ls $R/qml4-L2-ent-test-*.json | tail -1) \
  --ablation-val-run $(ls $R/qml4-L2-noent-val-*.json | tail -1) --ablation-test-run $(ls $R/qml4-L2-noent-test-*.json | tail -1)
```

Rebuild the report, the console and the deployable site:

```bash
python tools/build_report.py       # web/QURE_Lab_Research_Report.pdf
python tools/build_web_bundle.py   # web/index.html, web/qure_lab_standalone.html, web/qure_bundle.json
python tools/build_site.py         # site/ (console + data + PDF + AI function)
```

Add `--vr-url vr/` to `build_web_bundle.py` once a VR build exists, to show the "Open 3D lab" links.

---

## Deploy online

```bash
cd site
npx vercel deploy --prod
```

Then add `GEMINI_API_KEY` and `XAI_API_KEY` in Vercel under **Project → Settings → Environment Variables**, and deploy again. Full steps: `site/DEPLOY.md`.

---

## Project structure

```
src/quantum/        circuit, training, noise models, mitigation, export, CLI (run_experiment.py)
src/data/           LIDC builder (Role 1 cohort), Role 2 handoff converter
src/evaluation/     metrics, baselines, final reports (LIDC and Role 2)
data/role2/         primary cohort: angles, labels (train/val only), Role 2 originals
data/lidc/          replication cohort from LIDC reader contours
results/role2/      frozen models, runs, summary, test predictions for Role 2
results/            Role 1 LIDC cohort results and figures
schemas/            experiment record JSON schema
web/                console template, built pages, data bundle, report PDF, vendored jsPDF
server/             AI proxy (Gemini → Grok), local dev server, tests, deploy guide
site/               deployable build (generated by tools/build_site.py)
tools/              builders: report, bundle, site, assistant knowledge base, notebooks
notebooks/          Colab notebooks for the quantum model and noise study
tests/              Python tests
docs/               method, dataset card, protocol, claims, pitch, demo script, VR prompt, status
```

---

## Tests

| Suite | Command | Checks |
| --- | --- | --- |
| Quantum pipeline | `python -m pytest -q tests/` | Circuit matches Qiskit statevector · no patient leakage · no clipping · seeds reproducible · readout mitigation removes bias · schema-valid export · test split locked · both datasets meet the input contract |
| AI proxy | `cd site && npm test` | Gemini first · Grok fallback · both fail cleanly · questions are size-limited · only numbers from the case are sent · test labels never forwarded |

---

## Data, honesty and limits

- **Data.**
    - CIRDataset ([Zenodo 6762573](https://zenodo.org/records/6762573), CC BY 4.0), built on LIDC-IDRI (CC BY 3.0); please cite both.
    - Labels are radiologist suspicion ratings (1–2 low, 4–5 high; 3 excluded), **not confirmed cancer**.
- **Simulated only.** The noise settings are hypothetical, not taken from real hardware. A real IBM hardware run is future work.
- **Optimistic validation.** The threshold and restart were chosen on validation. Test results are pending with Role 2.
- **Not claimed:** quantum advantage, clinical validity, diagnosis.

Full list: `docs/CLAIMS_AND_LIMITATIONS.md`.

---

## Team and credits

| Role | Owns |
| --- | --- |
| Role 1 — Quantum | Circuit, training, noise, mitigation, evaluation, console |
| Role 2 — Data | CIRDataset preparation, features, split, held-out test labels, classical baseline |
| Role 3 — Web / VR | Console integration, 3D lab (`docs/PROMPT_5_VR_LAB.md`) |
| Role 4 — Pitch | Deck, demo script, claims review |

Built with [Qiskit](https://www.ibm.com/quantum/qiskit) 2.5.2 and Qiskit Aer 0.17.2. PDF export uses [jsPDF](https://github.com/parallax/jsPDF) (MIT).
