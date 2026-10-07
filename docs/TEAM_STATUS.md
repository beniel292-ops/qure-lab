# Team status — 2026-10-06 (updated after Role 2 merge)

**Role 2 delivered.** The CIRDataset cohort is now primary. Quantum models were trained on it and frozen. Validation is scored. 90 frozen test predictions are waiting for Role 2 to score them. The website was rebuilt as a console UI. See `docs/ROLE2_MERGE.md`.


PLANNED ≠ IMPLEMENTED ≠ TESTED ≠ CLINICALLY VALIDATED.

## Done and tested (in this package)

| Area | Status | Where |
| --- | --- | --- |
| Quantum classifier, noise, readout mitigation, export | TESTED (15 tests), run on both real cohorts | `src/quantum/`, `notebooks/02`, `notebooks/03` |
| Real dataset (Role 2 draft) | BUILT, leakage-checked, documented | `src/data/build_lidc_dataset.py`, `data/lidc/`, `docs/DATASET_CARD.md` |
| Classical baselines + metrics (Role 2 draft) | RUN on val and test | `src/evaluation/`, `results/metrics/` |
| Frozen protocol + final test run | DONE once | `docs/EXPERIMENT_PROTOCOL.md`, `results/final/` |
| Desktop dashboard (Role 3 reference) | BUILT, rendered in light, dark and phone views; published link | `web/`, `tools/build_web_bundle.py` |
| Claims checklist, pitch, demo script (Role 4 drafts) | WRITTEN from real numbers | `docs/` |

## Still to do (owner)

0. **Role 2 (now):** score `results/role2/final/test_predictions_for_role2.csv` against the held test labels. Score your logistic regression on the same 90 cases. Send back the metrics only. Role 1 then adds them to the site.

1. **Role 2:** review `DATASET_CARD.md`. Either accept the LIDC route or replace it. If you change the data, rerun everything (`docs/INTEGRATION.md`, "Full rerun").
2. **Role 3:** decide whether to adopt `web/` or merge its pieces into your own site. Optional: a Quest/WebXR view of the same bundle, only after desktop works.
3. **Role 4:** turn `PITCH.md` into slides. Have Role 1 review technical slides. Rehearse with `DEMO_SCRIPT.md`.
4. **Role 1:**
    - Run notebook 02 once inside Colab to confirm the setup cell (not yet tested in real Colab).
    - Optional stretch: ZNE, or a noise model from a backend calibration snapshot, as a separately labelled experiment.
5. **Everyone:**
    - Push to GitHub (branches below).
    - Choose the official demo surface: the published link or the standalone HTML.
    - Fill in unknowns: the deadline and the official scoring weights.

## Open risks

- Primary cohort: validation-only until Role 2 scores test. The quantum validation BA is optimistic (threshold and restart were chosen on validation).
- The pitch deck numbers come from the LIDC replication cohort. Update them to the primary cohort if the team agrees.

- Classical AUC (0.919) is higher than quantum AUC (0.882). Say this plainly if asked.
- The instability rule is evaluated on small counts (9 flagged at N4).
- Case-level confidence intervals ignore within-patient correlation.

## Update 2026-10-07: easier console, assistant, report

- The console opens on **Start**, which shows 3 common tasks and the key findings in plain words.
- **Simple / Detailed** switch in the top bar. Simple hides the dense tables, plus the Models, Data and Handoff pages.
- **Accuracy lab:** move the threshold, or switch between noiseless / noisy / mitigated and quantum / logistic regression. The confusion matrix and the accuracy formula update live (validation only).
- **Help & FAQ** page (about 25 answers plus a glossary). Dotted terms show definitions.
- **Ask QURE** assistant, in this order:
  1. built-in guide answers (offline);
  2. Gemini, then Grok, through `api/ask` (keys only in host env vars);
  3. Claude inside the claude.ai page;
  4. the closest guide answers if no AI is reachable.
- **Research report PDF** (`web/QURE_Lab_Research_Report.pdf`, built by `tools/build_report.py` from the same bundle).
- **Deployable site:** `python tools/build_site.py` creates `site/`. See `site/DEPLOY.md` for Vercel and the env vars.
- **To do:** deploy `site/` and add `GEMINI_API_KEY` / `XAI_API_KEY` in the host settings. Do not paste keys in chat or commit them.
