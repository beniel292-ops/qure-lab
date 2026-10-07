# Integration guide — how teammates' work joins this package

## How to give the whole project to the assistant (pick one)

1. **GitHub (best).** Push every teammate's work to `https://github.com/beniel292-ops/IBM-bin`, one branch per role (`quantum-core`, `data-evaluation`, `web-demo`, `pitch-docs`), then say "integrate the repo". The repo is public, so the assistant can clone it read-only. It cannot push; you push the result.
2. **Your connected IBM folder.** Copy each teammate's files into `/Users/user/Desktop/IBM/<role-name>/` (for example `role2-data/`, `role3-web/`, `role4-pitch/`), then say "integrate the IBM folder". The assistant can read the folder and write results back into it.
3. **Chat upload.** Zip each teammate's folder and attach the zips in this chat.

Include notebooks with their outputs (File → Download .ipynb from Colab), any CSV/JSON results, and slide files (.pptx or PDF).

## What the assistant will do on integration

1. Inspect each part. Nothing is overwritten without first diffing it against this package.
2. Check data contracts:
    - Role 2's `features.csv` and `features_meta.json` against `docs/QUANTUM_INPUT_CONTRACT.md`
    - Role 3's site against `schemas/experiment.schema.json` and the web bundle format
3. If Role 2's data differs from `data/lidc`, run the full rerun below, then rebuild the website bundle.
4. Run all tests, then rebuild and re-render the website in light, dark and phone views.
5. Re-check every number in the slides against `results/final/final_summary.json` and `CLAIMS_AND_LIMITATIONS.md`.

## Full rerun (any machine with Python 3.10+, or a Colab terminal)

```bash
pip install -r requirements-quantum.txt pylidc "setuptools<81" scikit-learn
python -m src.data.build_lidc_dataset --out data/lidc                     # ~4 min
python -m src.evaluation.baselines --split val
python -m src.quantum.run_experiment --features data/lidc/features.csv --meta data/lidc/features_meta.json --split val
python -m src.quantum.run_experiment --features data/lidc/features.csv --meta data/lidc/features_meta.json --split val --no-entangle
# freeze decisions (docs/EXPERIMENT_PROTOCOL.md), then ONCE:
python -m src.evaluation.baselines --split test --final
python -m src.quantum.run_experiment ... --split test --final --model results/models/<ent model>.json
python -m src.quantum.run_experiment ... --split test --final --model results/models/<noent model>.json
python -m src.evaluation.final_report --val-run results/runs/<ent val>.json --test-run results/runs/<ent test>.json --ablation-test-run results/runs/<noent test>.json
python tools/build_web_bundle.py --merged results/final/<ent test>-merged.json --summary results/final/final_summary.json
python -m pytest -q tests/
```

## Git

```bash
git clone https://github.com/beniel292-ops/IBM-bin && cd IBM-bin
git checkout -b quantum-core
# copy this package's contents into the repo root
git add . && git commit -m "QURE Lab: quantum pipeline, LIDC data, baselines, final results, dashboard, docs"
git push -u origin quantum-core
```

Then open a pull request to `main`. Never force-push.
