"""One command reproduces the Role 1 experiment end to end.

Development smoke test on a SYNTHETIC fixture (no real data needed):
    python -m src.quantum.run_experiment --fixture --split val --quick

Real data from Role 2 (validation first; test only for the final evaluation):
    python -m src.quantum.run_experiment --features data/features.csv --meta data/features_meta.json --split val
    python -m src.quantum.run_experiment --features ... --meta ... --split test --final --model results/models/<id>.json
"""
from __future__ import annotations

import argparse
import json
import os

from .config import DEFAULT_CAL_SHOTS, DEFAULT_LAYERS, DEFAULT_REPEATS, DEFAULT_SHOTS, NOISE_SETTINGS
from .data_io import load_dataset, make_fixture
from .execute import run_cases
from .export import build_records, new_run_id, validate_run_file, write_run
from .train import TrainedModel, train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="use a SYNTHETIC development fixture")
    ap.add_argument("--features")
    ap.add_argument("--meta")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to touch the TEST split")
    ap.add_argument("--model", help="frozen model JSON; if omitted, train a new one (train+val only)")
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--no-entangle", action="store_true", help="entangling-gate ablation")
    ap.add_argument("--settings", default=",".join(s["id"] for s in NOISE_SETTINGS))
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--cal-shots", type=int, default=DEFAULT_CAL_SHOTS)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="smoke test: fewer iterations/shots/repeats")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.split == "test" and not a.final:
        ap.error("the test split is held out: pass --final only for the final evaluation, with a frozen --model")
    if a.split == "test" and not a.model:
        ap.error("final test evaluation requires a frozen --model (no training on the final run)")
    if a.quick:
        a.maxiter, a.restarts, a.shots, a.cal_shots, a.repeats = 60, 2, 500, 500, 3

    if a.fixture:
        csv_path, meta_path = make_fixture(os.path.join(a.out, "fixtures", "input"))
    else:
        if not (a.features and a.meta):
            ap.error("--features and --meta are required unless --fixture")
        csv_path, meta_path = a.features, a.meta
    ds = load_dataset(csv_path, meta_path)
    print(f"data_status={ds.meta['data_status']}  cases={len(ds.case_id)}  "
          f"splits={ {s: ds.split.count(s) for s in ('train', 'val', 'test')} }")

    if a.model:
        model = TrainedModel.load(a.model)
        print(f"loaded frozen model {model.model_id} ({model.param_version})")
        if model.encoding != ds.encoding:
            ap.error(f"model encoding {model.encoding!r} != dataset input_encoding {ds.encoding!r}")
    else:
        _, Xtr, ytr = ds.subset("train")
        _, Xva, yva = ds.subset("val")
        model = train(Xtr, ytr, Xva, yva, layers=a.layers, entangle=not a.no_entangle,
                      restarts=a.restarts, maxiter=a.maxiter, data_status=ds.meta["data_status"],
                      encoding=ds.encoding)
        os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
        mpath = os.path.join(a.out, "models", f"{model.model_id}-{model.param_version}.json")
        model.save(mpath)
        print(f"saved frozen model -> {mpath}  (val balanced acc {model.val_balanced_accuracy:.3f}, "
              f"threshold {model.threshold})")

    idx, X, _ = ds.subset(a.split)
    wanted = set(a.settings.split(","))
    settings = [s for s in NOISE_SETTINGS if s["id"] in wanted]
    res = run_cases(X, model.theta, model.layers, model.entangling, settings=settings, shots=a.shots,
                    cal_shots=a.cal_shots, repeats=a.repeats, encoding=model.encoding)
    prefix = ("FIXTURE-" if ds.meta["data_status"] == "synthetic-fixture" else "") + f"{model.model_id}-{a.split}"
    run_id = new_run_id(prefix)
    recs = build_records(ds, idx, res, model, split=a.split, run_id=run_id)
    out_dir = os.path.join(a.out, "fixtures" if ds.meta["data_status"] == "synthetic-fixture" else "runs")
    w = write_run(out_dir, recs, model, ds, split=a.split, run_id=run_id,
                  config={"settings": [s["id"] for s in settings], "shots": a.shots, "cal_shots": a.cal_shots,
                          "repeats": a.repeats, "split": a.split, "final": a.final, "quick": a.quick})
    validate_run_file(w["path"])
    print(f"wrote {len(recs)} records -> {w['path']} (schema-valid)")
    print(json.dumps(w["summary"], indent=1))
    return w


if __name__ == "__main__":
    main()
