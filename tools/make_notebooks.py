"""Generate the Role 1 Colab notebooks (run once; output goes to notebooks/)."""
import nbformat as nbf

SETUP = r'''# --- Setup: works in Google Colab and locally -------------------------------------------
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/beniel292-ops/IBM-bin.git"
BRANCH = "quantum-core"          # change if the team merged to main

if IN_COLAB:
    if not os.path.exists("/content/IBM-bin"):
        r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, "/content/IBM-bin"])
        if r.returncode != 0:
            print("Clone failed (branch not pushed yet?). Upload role1_quantum.zip instead:")
            from google.colab import files
            up = files.upload()
            zipname = next(iter(up))
            subprocess.run(["unzip", "-q", "-o", zipname, "-d", "/content/IBM-bin"])
    os.chdir("/content/IBM-bin")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-quantum.txt", "scikit-learn"])
else:
    root = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
    os.chdir(root)
sys.path.insert(0, os.getcwd())
import qiskit, qiskit_aer
print("cwd:", os.getcwd(), "| qiskit", qiskit.__version__, "| aer", qiskit_aer.__version__)'''

DATA = r'''# --- Data source ---------------------------------------------------------------------------
# USE_FIXTURE = True  -> SYNTHETIC development fixture (NOT medical data, never evidence)
# USE_FIXTURE = False -> Role 2's real files (input contract: docs/QUANTUM_INPUT_CONTRACT.md)
USE_FIXTURE = False
REAL_FEATURES = "data/lidc/features.csv"
REAL_META = "data/lidc/features_meta.json"

from src.quantum import data_io
if USE_FIXTURE:
    csv_path, meta_path = data_io.make_fixture("results/fixtures/input")
else:
    csv_path, meta_path = REAL_FEATURES, REAL_META
ds = data_io.load_dataset(csv_path, meta_path)      # also checks patient leakage
print("data_status:", ds.meta["data_status"])
print("label definition:", ds.meta["label_definition"])
print("features:", ds.meta["feature_names"])
for s in ("train", "val", "test"):
    idx, X, y = ds.subset(s)
    print(f"{s:5s}: {len(idx):4d} cases, positives {int(y.sum())}, patients {len(set(ds.patient_group_id[i] for i in idx))}")
if ds.meta["data_status"] == "synthetic-fixture":
    print("\n*** SYNTHETIC FIXTURE: results below are for development only ***")'''

NB2 = [
    ("md", "# 02 · Quantum model (QURE Lab, Role 1)\n\nFour-qubit variational classifier in Qiskit. **Training uses only the train split; restart selection and the decision threshold use only the validation split. The test split is not touched in this notebook.**\n\nPipeline: four standardized features (from Role 2) → angle encoding `2·arctan(x)` → shallow RY + CX ansatz → measure all 4 qubits → `raw_score = P(odd parity)`.\n\n`raw_score` is measurement-derived. It is **not** a calibrated probability of the label."),
    ("code", SETUP),
    ("code", DATA),
    ("md", "## The circuit\nOne case, with real angles. `transpiled_ops` are counted in the basis `rz, sx, x, cx` at optimization level 0."),
    ("code", r'''from src.quantum import model
import numpy as np, json
info = model.circuit_info(layers=2, entangle=True)
print(info["text_drawing"])
print(json.dumps({k: v for k, v in info.items() if k != "text_drawing"}, indent=1))'''),
    ("md", "## Correctness check: fast exact path == Qiskit Statevector\nTraining uses a fast exact evaluator built from Qiskit's `Operator`. This cell proves it matches the full Qiskit circuit."),
    ("code", r'''rng = np.random.default_rng(0)
th = rng.uniform(0, 2*np.pi, model.n_params(2))
_, Xva, yva = ds.subset("val")
fast = model.exact_scores(Xva[:10], th, 2, True)
slow = np.array([model.statevector_score(x, th, 2, True) for x in Xva[:10]])
print("max |fast - statevector| =", float(np.max(np.abs(fast - slow))))
assert np.allclose(fast, slow, atol=1e-10)'''),
    ("md", "## Train (noiseless, exact) and freeze\nClass-balanced binary cross-entropy, SciPy COBYLA, 3 seeded restarts. The best restart is chosen by **validation** loss, and the threshold by validation balanced accuracy."),
    ("code", r'''from src.quantum.train import train
_, Xtr, ytr = ds.subset("train")
_, Xva, yva = ds.subset("val")
os.makedirs("results/models", exist_ok=True)
m_ent = train(Xtr, ytr, Xva, yva, layers=2, entangle=True, restarts=3, maxiter=200,
              data_status=ds.meta["data_status"])
p_ent = f"results/models/{m_ent.model_id}-{m_ent.param_version}.json"; m_ent.save(p_ent)
print("saved", p_ent, "| val balanced acc", round(m_ent.val_balanced_accuracy, 3), "| threshold", m_ent.threshold)'''),
    ("md", "## Ablation: same circuit without entangling gates\nSame parameters, same observable, no CX gates. We do **not** assume entanglement helps; this cell measures it."),
    ("code", r'''m_noent = train(Xtr, ytr, Xva, yva, layers=2, entangle=False, restarts=3, maxiter=200,
                data_status=ds.meta["data_status"])
p_noent = f"results/models/{m_noent.model_id}-{m_noent.param_version}.json"; m_noent.save(p_noent)
print("entangled   : val balanced acc", round(m_ent.val_balanced_accuracy, 3))
print("no entangle : val balanced acc", round(m_noent.val_balanced_accuracy, 3))'''),
    ("code", r'''import matplotlib.pyplot as plt
s = model.exact_scores(Xva, m_ent.theta, 2, True)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(s[yva == 0], bins=15, alpha=.6, label="label 0")
ax.hist(s[yva == 1], bins=15, alpha=.6, label="label 1")
ax.axvline(m_ent.threshold, color="k", ls="--", label=f"threshold {m_ent.threshold}")
ax.set_xlabel("raw_score (noiseless, validation)"); ax.set_ylabel("cases"); ax.legend()
ax.set_title("Validation raw scores" + (" — SYNTHETIC FIXTURE" if ds.meta["data_status"] != "real" else ""))
plt.tight_layout(); plt.show()'''),
    ("md", "**Next:** notebook 03 loads the frozen model file printed above and runs the noise and mitigation experiments."),
]

NB3 = [
    ("md", "# 03 · Noise and readout mitigation (QURE Lab, Role 1)\n\nA frozen model is executed under documented **hypothetical** simulator noise settings N0–N4. Each case and setting is run with several seeded repeats. For every case we report:\n\n- **noiseless reference** (exact)\n- **noisy** finite-shot scores\n- **noisy at an equal total shot budget** (fair comparison with mitigation)\n- **readout-mitigated** scores, raw and never clipped\n- **flip rates** against the noiseless decision\n\nMitigation targets **readout error only**. It does not undo gate noise, and it can increase variability."),
    ("code", SETUP),
    ("code", DATA),
    ("code", r'''import glob, json
from src.quantum.config import NOISE_SETTINGS, NOISE_SOURCE, MITIGATION
from src.quantum.train import TrainedModel
FROZEN = "results/models/qml4-L2-ent-20261006-153140.json"   # the frozen model used for the final test run
MODEL_PATH = FROZEN if os.path.exists(FROZEN) else sorted(glob.glob("results/models/*-ent-*.json"))[-1]
mdl = TrainedModel.load(MODEL_PATH)
print("frozen model:", MODEL_PATH, "| threshold", mdl.threshold)
print("noise source:", NOISE_SOURCE)
for s in NOISE_SETTINGS: print(" ", s)
print(json.dumps(MITIGATION, indent=1))'''),
    ("md", "## Run on the validation split\nThe validation split is used to look at results and to design any instability rule. The test split runs only in the final, guarded cell at the end."),
    ("code", r'''from src.quantum.execute import run_cases
from src.quantum.export import build_records, write_run, new_run_id, validate_run_file
SHOTS, CAL_SHOTS, REPEATS = 2000, 2000, 5
idx, Xva, yva = ds.subset("val")
res = run_cases(Xva, mdl.theta, mdl.layers, mdl.entangling, shots=SHOTS, cal_shots=CAL_SHOTS, repeats=REPEATS)
prefix = ("FIXTURE-" if ds.meta["data_status"] == "synthetic-fixture" else "") + f"{mdl.model_id}-val"
run_id = new_run_id(prefix)
recs = build_records(ds, idx, res, mdl, split="val", run_id=run_id)
out_dir = "results/fixtures" if ds.meta["data_status"] == "synthetic-fixture" else "results/runs"
w = write_run(out_dir, recs, mdl, ds, split="val", run_id=run_id,
              config={"shots": SHOTS, "cal_shots": CAL_SHOTS, "repeats": REPEATS, "split": "val"})
validate_run_file(w["path"]); print("wrote", w["path"], "(schema-valid)")'''),
    ("code", r'''import pandas as pd
summary = pd.DataFrame(w["summary"]).set_index("setting_id")
summary'''),
    ("code", r'''import matplotlib.pyplot as plt, numpy as np
tag = " — SYNTHETIC FIXTURE" if ds.meta["data_status"] != "real" else ""
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
x = np.arange(len(summary))
axs[0].bar(x-.2, summary.mean_abs_dev_noisy, .4, label="noisy"); axs[0].bar(x+.2, summary.mean_abs_dev_mitigated, .4, label="mitigated")
axs[0].set_title("mean |score - noiseless|" + tag)
axs[1].bar(x-.2, summary.mean_flip_rate_noisy, .4, label="noisy"); axs[1].bar(x+.2, summary.mean_flip_rate_mitigated, .4, label="mitigated")
axs[1].set_title("decision flip rate vs noiseless")
axs[2].bar(x-.27, summary.mean_std_noisy, .27, label="noisy"); axs[2].bar(x, summary.mean_std_mitigated, .27, label="mitigated")
axs[2].bar(x+.27, summary.mean_std_noisy_equal_budget, .27, label="noisy, equal shot budget")
axs[2].set_title("repeat-to-repeat std")
for a in axs: a.set_xticks(x, summary.index); a.legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
    ("md", "## Every case, not cherry-picked\nThe table lists all validation cases at one setting, including cases where mitigation moved the score **farther** from the noiseless reference, and cases where the noiseless model is **wrong** against the label."),
    ("code", r'''SETTING = "N4"
rows = []
for r in recs:
    if r["noise"]["setting_id"] != SETTING: continue
    o = r["outputs"]
    rows.append({"case": r["case_id"], "label": r["reference_label"],
                 "noiseless": o["noiseless_exact"]["raw_score"], "noisy_mean": o["noisy"]["mean"],
                 "mitigated_mean": o["mitigated"]["mean"], "flip_noisy": o["flip_rate_noisy_vs_noiseless"],
                 "flip_mit": o["flip_rate_mitigated_vs_noiseless"],
                 "mitigation_closer": o["abs_dev_mitigated_mean"] < o["abs_dev_noisy_mean"],
                 "noiseless_correct": o["noiseless_correct_vs_label"]})
pd.DataFrame(rows)'''),
    ("md", "## FINAL test evaluation (run once, at the end)\nThe final test run for the frozen model **has already been done once** (`results/final/`). Keep `FINAL = False`. Re-running test after seeing results would break the protocol, unless you report it as a new, separately labelled experiment."),
    ("code", r'''FINAL = False
if FINAL:
    idx_t, Xte, yte = ds.subset("test")
    res_t = run_cases(Xte, mdl.theta, mdl.layers, mdl.entangling, shots=SHOTS, cal_shots=CAL_SHOTS, repeats=REPEATS)
    rid = new_run_id(("FIXTURE-" if ds.meta["data_status"] == "synthetic-fixture" else "") + f"{mdl.model_id}-test")
    recs_t = build_records(ds, idx_t, res_t, mdl, split="test", run_id=rid)
    wt = write_run(out_dir, recs_t, mdl, ds, split="test", run_id=rid,
                   config={"shots": SHOTS, "cal_shots": CAL_SHOTS, "repeats": REPEATS, "split": "test", "final": True})
    validate_run_file(wt["path"]); print("FINAL test run:", wt["path"])
    display(pd.DataFrame(wt["summary"]).set_index("setting_id"))
else:
    print("Test split untouched (FINAL = False).")'''),
]


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["colab"] = {"provenance": []}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    nbf.write(nb, path)


build(NB2, "notebooks/02_quantum_model.ipynb")
build(NB3, "notebooks/03_noise_and_mitigation.ipynb")
print("notebooks written")
