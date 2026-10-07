"""Turn execution results into versioned experiment records (schemas/experiment.schema.json)."""
from __future__ import annotations

import json
import os
import platform
import time
import uuid

import numpy as np
import qiskit
import qiskit_aer

from .config import ENCODING, MITIGATION, NOISE_SOURCE, SCHEMA_VERSION
from .model import circuit_info, encode_angles

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "schemas", "experiment.schema.json")


def _stats(v: np.ndarray) -> dict:
    return {"values": [round(float(x), 6) for x in v], "mean": round(float(np.mean(v)), 6),
            "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 6)}


def build_records(ds, idx, results: dict, model, *, split: str, run_id: str) -> list:
    info = circuit_info(model.layers, model.entangling)
    info_short = {k: v for k, v in info.items() if k != "text_drawing"}
    thr = model.threshold
    recs = []
    for sid, R in results.items():
        s = R["setting"]
        for j, i in enumerate(idx):
            y = int(ds.label[i]) if int(ds.label[i]) in (0, 1) else None  # None = label held back (e.g. test)
            ref = float(R["noiseless_exact"][j])
            d_ref = int(ref >= thr)
            noisy, mit, eq = R["noisy"][:, j], R["mitigated"][:, j], R["noisy_equal_budget"][:, j]
            d_noisy = (noisy >= thr).astype(int)
            d_mit = (mit >= thr).astype(int)
            rec = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "data_status": ds.meta["data_status"],
                "result_type": "simulated",
                "simulator": f"qiskit-aer {qiskit_aer.__version__}",
                "case_id": ds.case_id[i],
                "patient_group_id": ds.patient_group_id[i],
                "split": split,
                "dataset": {k: ds.meta[k] for k in ("dataset_source", "dataset_version", "label_definition",
                                                    "label_positive_meaning")},
                "reference_label": y,
                "features": {"names": ds.meta["feature_names"],
                             "values": [round(float(v), 6) for v in ds.X[i]],
                             "preprocessing_version": ds.meta["preprocessing_version"]},
                "encoding": {"method": ENCODING if model.encoding == "arctan" else
                             "precomputed angles supplied by Role 2 (theta = pi*sigmoid(train-standardized feature)), used as-is",
                             "angles_rad": [round(float(a), 6) for a in encode_angles(ds.X[i:i + 1], model.encoding)[0]]},
                "model": {"model_id": model.model_id, "param_version": model.param_version,
                          "layers": model.layers, "entangling": model.entangling,
                          "n_trainable_params": len(model.theta), "trained_on": model.trained_on},
                "circuit": info_short,
                "noise": {"setting_id": sid, "label": s["label"],
                          "params": {k: s[k] for k in ("p1q", "p2q", "readout_p10", "readout_p01")},
                          "source": NOISE_SOURCE},
                "mitigation": {**MITIGATION, "calibration_shots_total": R["budget"]["calibration_shots_total"],
                               "calibration_estimates": R["calibrations"]},
                "execution": {"shots": R["budget"]["shots_per_execution"], "repeats": len(noisy),
                              "seeds": R["seeds"], "setting_runtime_s_all_cases": R["runtime_s"],
                              "budget": R["budget"]},
                "outputs": {
                    "score_definition": "raw_score = P(odd parity of the 4 measured bits); NOT a calibrated probability",
                    "decision_threshold": thr,
                    "threshold_source": "validation split, max balanced accuracy",
                    "noiseless_exact": {"raw_score": round(ref, 6), "decision": d_ref},
                    "noisy_exact": {"raw_score": round(float(R["noisy_exact"][j]), 6),
                                    "note": "infinite-shot noisy expectation (gate noise via density matrix, readout analytic)"},
                    "noisy": {**_stats(noisy), "decisions": d_noisy.tolist()},
                    "noisy_equal_budget": {**_stats(eq), "shots": R["budget"]["equal_budget_raw_shots"]},
                    "mitigated": {**_stats(mit), "decisions": d_mit.tolist(),
                                  "negative_quasiprob_repeats": int(R["negative_quasiprob"][:, j].sum()),
                                  "score_out_of_range_repeats": int(R["out_of_range"][:, j].sum()),
                                  "note": "raw mitigated estimates; values outside [0,1] are kept, not clipped"},
                    "flip_rate_noisy_vs_noiseless": round(float(np.mean(d_noisy != d_ref)), 6),
                    "flip_rate_mitigated_vs_noiseless": round(float(np.mean(d_mit != d_ref)), 6),
                    "abs_dev_noisy_mean": round(abs(float(np.mean(noisy)) - ref), 6),
                    "abs_dev_mitigated_mean": round(abs(float(np.mean(mit)) - ref), 6),
                    "noiseless_correct_vs_label": None if y is None else bool(d_ref == y),
                },
                "calibrated_probability": None,
                "classical_baseline": None,
                "state_diagnostics": None if R["state_noiseless"] is None else {
                    "note": "per-qubit Bloch vectors of reduced states; entanglement alone shortens them; "
                            "NOT a confidence or reliability measure; readout mitigation does not change the state",
                    "noiseless": R["state_noiseless"][j], "gate_noisy": R["state_noisy_gate"][j]},
            }
            recs.append(rec)
    return recs


def summarize(records: list) -> list:
    """Quantum-side reliability summary per noise setting. Role 2 owns the final evaluation metrics."""
    by = {}
    for r in records:
        by.setdefault(r["noise"]["setting_id"], []).append(r)
    rows = []
    for sid, rs in by.items():
        o = [r["outputs"] for r in rs]
        y = np.array([-1 if r["reference_label"] is None else r["reference_label"] for r in rs])
        def ba(pred):
            if not np.any(y >= 0):
                return None  # labels held back (e.g. test split before scoring by Role 2)
            tpr = np.mean(pred[y == 1] == 1) if np.any(y == 1) else np.nan
            tnr = np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan
            return round(float(np.nanmean([tpr, tnr])), 4)
        ref_pred = np.array([x["noiseless_exact"]["decision"] for x in o])
        noisy_first = np.array([x["noisy"]["decisions"][0] for x in o])
        mit_first = np.array([x["mitigated"]["decisions"][0] for x in o])
        dev_n = np.array([x["abs_dev_noisy_mean"] for x in o])
        dev_m = np.array([x["abs_dev_mitigated_mean"] for x in o])
        rows.append({
            "setting_id": sid, "label": rs[0]["noise"]["label"], "n_cases": len(rs),
            "balanced_accuracy_noiseless": ba(ref_pred),
            "balanced_accuracy_noisy_repeat0": ba(noisy_first),
            "balanced_accuracy_mitigated_repeat0": ba(mit_first),
            "mean_flip_rate_noisy": round(float(np.mean([x["flip_rate_noisy_vs_noiseless"] for x in o])), 4),
            "mean_flip_rate_mitigated": round(float(np.mean([x["flip_rate_mitigated_vs_noiseless"] for x in o])), 4),
            "mean_abs_dev_noisy": round(float(dev_n.mean()), 5),
            "mean_abs_dev_mitigated": round(float(dev_m.mean()), 5),
            "frac_cases_mitigation_closer": round(float(np.mean(dev_m < dev_n)), 4),
            "frac_cases_mitigation_farther": round(float(np.mean(dev_m > dev_n)), 4),
            "mean_std_noisy": round(float(np.mean([x["noisy"]["std"] for x in o])), 5),
            "mean_std_mitigated": round(float(np.mean([x["mitigated"]["std"] for x in o])), 5),
            "mean_std_noisy_equal_budget": round(float(np.mean([x["noisy_equal_budget"]["std"] for x in o])), 5),
            "negative_quasiprob_repeats_total": int(sum(x["mitigated"]["negative_quasiprob_repeats"] for x in o)),
            "score_out_of_range_repeats_total": int(sum(x["mitigated"]["score_out_of_range_repeats"] for x in o)),
        })
    return rows


def write_run(out_dir: str, records: list, model, ds, *, split: str, run_id: str, config: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    info = circuit_info(model.layers, model.entangling)
    run = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_status": ds.meta["data_status"], "result_type": "simulated", "split": split,
        "environment": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                        "python": platform.python_version()},
        "config": config, "model": {k: v for k, v in vars(model).items()},
        "circuit": info, "summary_quantum_side": summarize(records),
        "warning": ("SYNTHETIC FIXTURE - development only, not evidence"
                    if ds.meta["data_status"] == "synthetic-fixture" else None),
        "records": records,
    }
    path = os.path.join(out_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(run, f, indent=1)
    return {"path": path, "summary": run["summary_quantum_side"]}


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def validate_run_file(path: str) -> None:
    import jsonschema
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    with open(path) as f:
        run = json.load(f)
    for rec in run["records"]:
        jsonschema.validate(rec, schema)
