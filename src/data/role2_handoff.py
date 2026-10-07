"""Convert the Role 2 handoff (CIRDataset, Zenodo 6762573) into the Role 1 input contract.

Role 2 delivers precomputed rotation angles theta = pi * sigmoid((feature - train_mean) / train_scale),
with train-only statistics, a patient-level split, and NO test labels (held back until models and
thresholds are frozen). This script does not compute or infer any labels; it only reshapes files.

    python -m src.data.role2_handoff --src <extracted handoff dir> --out data/role2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

from src.quantum.config import INPUT_CONTRACT_VERSION

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
COPY = ["DATASET_CARD.json", "preprocessing.json", "classical_model.json", "validation_metrics.json",
        "validation_interval.json", "validation_classical_predictions.csv", "linked_case.json",
        "linked_validation_case.png", "validation_charts.png", "split_manifest.csv", "environment.json",
        "README.txt"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/role2")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "role2_original"), exist_ok=True)

    card = json.load(open(os.path.join(a.src, "DATASET_CARD.json")))
    prep = json.load(open(os.path.join(a.src, "preprocessing.json")))
    rows = []
    for fname, split in [("train_quantum_inputs.csv", "train"), ("validation_quantum_inputs.csv", "validation"),
                         ("test_quantum_inputs.csv", "test")]:
        with open(os.path.join(a.src, fname), newline="") as f:
            for r in csv.DictReader(f):
                label = r.get("target", "")
                if split == "test" and label not in ("", None):
                    raise ValueError("unexpected test label in Role 2 handoff; refusing to use it")
                rows.append({"case_id": r["case_id"], "patient_group_id": r["patient_id"],
                             "split": SPLIT_MAP[split], "label": label or "",
                             **{f"theta_{j}": r[f"theta_{j}"] for j in range(4)}})
    # cross-check against Role 2's split manifest
    manifest = {r["case_id"]: SPLIT_MAP[r["split"]] for r in csv.DictReader(open(os.path.join(a.src, "split_manifest.csv")))}
    bad = [r["case_id"] for r in rows if manifest.get(r["case_id"]) != r["split"]]
    if bad or len(manifest) != len(rows):
        raise ValueError(f"split manifest mismatch: {bad[:5]} (rows {len(rows)} vs manifest {len(manifest)})")

    with open(os.path.join(a.out, "features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "patient_group_id", "split", "label",
                                          "theta_0", "theta_1", "theta_2", "theta_3"])
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    meta = {
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "data_status": "real",
        "dataset_source": f"CIRDataset (Zenodo record 6762573, {card['source_license']}); "
                          "LIDC-IDRI CT patches with expert nodule masks. Prepared by Role 2.",
        "dataset_version": f"{card['archive']} {card['archive_md5']}",
        "label_definition": card["target"],
        "label_positive_meaning": "high radiologist suspicion (rating 4-5); NOT pathology-confirmed cancer",
        "feature_names": ["theta_0", "theta_1", "theta_2", "theta_3"],
        "feature_sources": prep["features"],
        "input_encoding": "precomputed_angles",
        "preprocessing_version": "role2-pi-sigmoid-train-only-v1",
        "preprocessing_note": prep["mapping"] + " (train-split mean/scale only; angles in (0, pi))",
        "unit": card["unit"],
        "split_counts": counts,
        "test_labels": "held by Role 2 until models and thresholds are frozen",
        "role2_status": card["status"],
        "limitations": card["limitations"],
    }
    json.dump(meta, open(os.path.join(a.out, "features_meta.json"), "w"), indent=2)
    for c in COPY:
        p = os.path.join(a.src, c)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(a.out, "role2_original", c))
    print(f"wrote {len(rows)} cases {counts} -> {a.out}")


if __name__ == "__main__":
    main()
