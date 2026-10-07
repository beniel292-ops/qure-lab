"""Input contract between Role 2 (data) and Role 1 (quantum).

Role 2 delivers two files:
  features.csv        columns: case_id, patient_group_id, split, label, <4 feature columns>
  features_meta.json  see docs/QUANTUM_INPUT_CONTRACT.md

Until real data exists, make_fixture() writes a SYNTHETIC development fixture. Every
record derived from it carries data_status = "synthetic-fixture" and must never be
presented as evidence.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import INPUT_CONTRACT_VERSION

REQUIRED_META = ["input_contract_version", "data_status", "dataset_source", "dataset_version",
                 "label_definition", "label_positive_meaning", "feature_names",
                 "preprocessing_version", "preprocessing_note"]
SPLITS = ("train", "val", "test")


@dataclass
class Dataset:
    meta: dict
    case_id: list = field(default_factory=list)
    patient_group_id: list = field(default_factory=list)
    split: list = field(default_factory=list)
    label: np.ndarray = None
    X: np.ndarray = None  # shape (n, 4): standardized features, or precomputed angles (see encoding)

    @property
    def encoding(self) -> str:
        return self.meta.get("input_encoding", "arctan")

    def subset(self, which: str):
        idx = [i for i, s in enumerate(self.split) if s == which]
        return (np.asarray(idx), self.X[idx], self.label[idx])


def check_no_patient_leakage(patient_group_id, split) -> None:
    """Raise if any patient group appears in more than one split."""
    seen: dict = {}
    for p, s in zip(patient_group_id, split):
        seen.setdefault(p, set()).add(s)
    leaked = {p: sorted(s) for p, s in seen.items() if len(s) > 1}
    if leaked:
        sample = list(leaked.items())[:5]
        raise ValueError(f"PATIENT LEAKAGE: {len(leaked)} patient groups span several splits, e.g. {sample}")


def load_dataset(csv_path: str, meta_path: str) -> Dataset:
    with open(meta_path) as f:
        meta = json.load(f)
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"features_meta.json missing keys: {missing}")
    if meta["input_contract_version"] != INPUT_CONTRACT_VERSION:
        raise ValueError(f"input contract {meta['input_contract_version']} != {INPUT_CONTRACT_VERSION}")
    names = meta["feature_names"]
    if len(names) != 4:
        raise ValueError("exactly 4 feature_names are required")
    encoding = meta.get("input_encoding", "arctan")
    if encoding not in ("arctan", "precomputed_angles"):
        raise ValueError(f"unknown input_encoding {encoding!r}")

    ds = Dataset(meta=meta)
    rows_X, rows_y = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"case_id", "patient_group_id", "split", "label", *names}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f"features.csv must have columns {sorted(need)}")
        for r in reader:
            if r["split"] not in SPLITS:
                raise ValueError(f"bad split {r['split']!r} for case {r['case_id']}")
            lab = (r["label"] or "").strip()
            if lab == "" and r["split"] == "test":
                y = -1  # test labels held back by the data owner until models/thresholds are frozen
            else:
                y = int(lab)
                if y not in (0, 1):
                    raise ValueError(f"label must be 0/1 (case {r['case_id']}); blank allowed only for test")
            x = [float(r[n]) for n in names]
            if not all(math.isfinite(v) for v in x):
                raise ValueError(f"non-finite feature in case {r['case_id']}")
            if encoding == "precomputed_angles" and not all(0.0 <= v <= math.pi for v in x):
                raise ValueError(f"precomputed angle outside [0, pi] in case {r['case_id']}")
            ds.case_id.append(r["case_id"])
            ds.patient_group_id.append(r["patient_group_id"])
            ds.split.append(r["split"])
            rows_X.append(x)
            rows_y.append(y)
    if len(set(ds.case_id)) != len(ds.case_id):
        raise ValueError("duplicate case_id values")
    check_no_patient_leakage(ds.patient_group_id, ds.split)
    ds.X = np.asarray(rows_X, dtype=float)
    ds.label = np.asarray(rows_y, dtype=int)
    for s in ("train", "val"):
        if s not in ds.split:
            raise ValueError(f"split '{s}' is empty")
    return ds


def make_fixture(out_dir: str, n_patients: int = 90, seed: int = 7) -> tuple[str, str]:
    """Write a SYNTHETIC fixture (development only). Patient-level split, 1-3 cases per patient."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    patients = [f"FXP{i:03d}" for i in range(n_patients)]
    order = rng.permutation(n_patients)
    n_tr, n_va = int(0.6 * n_patients), int(0.2 * n_patients)
    split_of = {}
    for k, i in enumerate(order):
        split_of[patients[i]] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    names = ["fixture_feat_a", "fixture_feat_b", "fixture_feat_c", "fixture_feat_d"]
    rows = []
    c = 0
    for p in patients:
        for _ in range(int(rng.integers(1, 4))):
            y = int(rng.random() < 0.4)
            mu = np.array([0.6, -0.4, 0.3, 0.2]) * (1 if y else -1)
            x = mu + rng.normal(0, 1.0, 4)
            rows.append({"case_id": f"FXC{c:04d}", "patient_group_id": p, "split": split_of[p],
                         "label": y, **{n: round(float(v), 5) for n, v in zip(names, x)}})
            c += 1
    # standardize with TRAIN statistics only (mirrors what Role 2 must do)
    tr = np.array([[r[n] for n in names] for r in rows if r["split"] == "train"])
    mu, sd = tr.mean(0), tr.std(0)
    for r in rows:
        for j, n in enumerate(names):
            r[n] = round((r[n] - mu[j]) / sd[j], 6)
    csv_path = os.path.join(out_dir, "features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "patient_group_id", "split", "label", *names])
        w.writeheader()
        w.writerows(rows)
    meta = {
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "data_status": "synthetic-fixture",
        "dataset_source": "SYNTHETIC FIXTURE generated by src/quantum/data_io.py:make_fixture (not medical data)",
        "dataset_version": f"fixture-seed{seed}",
        "label_definition": "SYNTHETIC class label (no medical meaning)",
        "label_positive_meaning": "synthetic class 1",
        "feature_names": names,
        "preprocessing_version": "fixture-standardize-train-only-v1",
        "preprocessing_note": "z-score using train-split mean/std only",
    }
    meta_path = os.path.join(out_dir, "features_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return csv_path, meta_path
