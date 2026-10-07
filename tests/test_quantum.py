"""Role 1 validation tests. Run:  python -m pytest -q tests/"""
import inspect
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum import config, data_io, execute, export, model, train  # noqa: E402


@pytest.fixture(scope="module")
def fixture_ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    c, m = data_io.make_fixture(str(d), n_patients=40)
    return data_io.load_dataset(c, m)


def test_fast_path_equals_qiskit_statevector():
    rng = np.random.default_rng(1)
    for ent in (True, False):
        th = rng.uniform(0, 2 * np.pi, model.n_params(2))
        X = rng.normal(0, 1.5, (6, 4))
        fast = model.exact_scores(X, th, 2, ent)
        slow = [model.statevector_score(x, th, 2, ent) for x in X]
        assert np.allclose(fast, slow, atol=1e-10)


def test_encoding_is_bounded_and_unclipped():
    a = model.encode_angles(np.array([[-1e6, -1.0, 0.0, 1e6]]))
    assert np.all(np.abs(a) < np.pi) and a[0, 2] == 0.0
    assert np.all(np.diff(model.encode_angles(np.linspace(-5, 5, 50)[:, None]).ravel()) > 0)


def test_no_entanglement_ablation_has_no_cx():
    assert "cx" not in model.circuit_info(2, False)["transpiled_ops"]
    assert model.circuit_info(2, True)["transpiled_ops"]["cx"] == 6


def test_patient_leakage_detected():
    with pytest.raises(ValueError, match="LEAKAGE"):
        data_io.check_no_patient_leakage(["p1", "p1", "p2"], ["train", "test", "val"])


def test_fixture_is_labelled_and_leak_free(fixture_ds):
    assert fixture_ds.meta["data_status"] == "synthetic-fixture"
    data_io.check_no_patient_leakage(fixture_ds.patient_group_id, fixture_ds.split)


def test_train_api_cannot_see_test():
    params = inspect.signature(train.train).parameters
    assert {"X_tr", "y_tr", "X_va", "y_va"} <= set(params)
    assert not any("test" in p for p in params)


def test_zero_noise_consistency_and_seed_reproducibility(fixture_ds):
    idx, X, y = fixture_ds.subset("val")
    th = np.random.default_rng(3).uniform(0, 2 * np.pi, model.n_params(2))
    n0 = [config.setting_by_id("N0")]
    r1 = execute.run_cases(X[:6], th, 2, True, settings=n0, shots=400, cal_shots=400, repeats=2, log=lambda *_: None)
    r2 = execute.run_cases(X[:6], th, 2, True, settings=n0, shots=400, cal_shots=400, repeats=2, log=lambda *_: None)
    R = r1["N0"]
    assert np.allclose(R["noisy_exact"], R["noiseless_exact"], atol=1e-10)  # zero noise == noiseless
    assert np.array_equal(R["noisy"], r2["N0"]["noisy"])                     # same seeds -> same counts
    assert np.array_equal(R["mitigated"], R["noisy"])                         # identity calibration at N0


def test_readout_mitigation_removes_readout_bias():
    """Readout-only noise: mitigated mean must be statistically consistent with noiseless."""
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (8, 4))
    th = rng.uniform(0, 2 * np.pi, model.n_params(2))
    R = execute.run_cases(X, th, 2, True, settings=[config.setting_by_id("N1")], shots=20000,
                          cal_shots=20000, repeats=3, with_state_diagnostics=False, log=lambda *_: None)["N1"]
    ref = R["noiseless_exact"]
    bias_noisy = np.abs(R["noisy"].mean(0) - ref).mean()
    bias_mit = np.abs(R["mitigated"].mean(0) - ref).mean()
    assert bias_mit < 0.5 * bias_noisy
    assert bias_mit < 0.01


def test_tensored_inverse_is_exact_without_sampling():
    s = config.setting_by_id("N4")
    M = execute.tensored([execute.readout_matrix(s["readout_p10"], s["readout_p01"])] * 4)
    p = np.random.default_rng(0).dirichlet(np.ones(16))
    cal = {"inverse": np.linalg.inv(M)}
    score, negq, oor = execute.mitigate(M @ p, cal)
    assert abs(score - execute.score_from_probs(p)) < 1e-12 and not oor


def test_mitigated_scores_are_not_clipped():
    src = inspect.getsource(execute.mitigate)
    assert "np.clip" not in src and "min(" not in src and "max(" not in src


def test_export_schema_valid(fixture_ds, tmp_path):
    _, Xtr, ytr = fixture_ds.subset("train")
    idx, Xva, yva = fixture_ds.subset("val")
    m = train.train(Xtr, ytr, Xva, yva, restarts=1, maxiter=20, data_status="synthetic-fixture", log=lambda *_: None)
    res = execute.run_cases(Xva[:5], m.theta, m.layers, m.entangling,
                            settings=[config.setting_by_id("N0"), config.setting_by_id("N3")],
                            shots=300, cal_shots=300, repeats=2, log=lambda *_: None)
    recs = export.build_records(fixture_ds, idx[:5], res, m, split="val", run_id="TEST-run")
    w = export.write_run(str(tmp_path), recs, m, fixture_ds, split="val", run_id="TEST-run", config={})
    export.validate_run_file(w["path"])
    run = json.load(open(w["path"]))
    assert run["warning"].startswith("SYNTHETIC")
    assert all(r["data_status"] == "synthetic-fixture" for r in run["records"])
    assert all(r["result_type"] == "simulated" for r in run["records"])
    assert all(r["calibrated_probability"] is None for r in run["records"])  # never faked


def test_cli_refuses_test_split_without_final():
    from src.quantum import run_experiment
    with pytest.raises(SystemExit):
        run_experiment.main(["--fixture", "--split", "test"])


def test_real_lidc_dataset_contract():
    """The shipped real dataset satisfies the input contract and is patient-leak-free."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_p, meta_p = os.path.join(root, "data/lidc/features.csv"), os.path.join(root, "data/lidc/features_meta.json")
    if not os.path.exists(csv_p):
        pytest.skip("data/lidc not built")
    ds = data_io.load_dataset(csv_p, meta_p)  # raises on leakage / bad values
    assert ds.meta["data_status"] == "real"
    assert len(ds.case_id) == sum(v["nodules"] for v in ds.meta["split_counts"].values())
    tr = ds.subset("train")[1]
    assert np.allclose(tr.mean(0), 0, atol=1e-6) and np.allclose(tr.std(0), 1, atol=1e-3)  # train-only scaling


def test_precomputed_angles_path_matches_qiskit():
    rng = np.random.default_rng(4)
    th = rng.uniform(0, 2 * np.pi, model.n_params(2))
    A = rng.uniform(0, np.pi, (5, 4))
    assert np.array_equal(model.encode_angles(A, "precomputed_angles"), A)
    fast = model.exact_scores(A, th, 2, True, "precomputed_angles")
    slow = [model.statevector_score(a, th, 2, True, "precomputed_angles") for a in A]
    assert np.allclose(fast, slow, atol=1e-10)


def test_role2_dataset_contract():
    """Role 2 handoff: precomputed angles in [0, pi], leak-free, test labels withheld (never invented)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_p, meta_p = os.path.join(root, "data/role2/features.csv"), os.path.join(root, "data/role2/features_meta.json")
    if not os.path.exists(csv_p):
        pytest.skip("data/role2 not built")
    ds = data_io.load_dataset(csv_p, meta_p)
    assert ds.encoding == "precomputed_angles" and ds.meta["data_status"] == "real"
    assert np.all((ds.X >= 0) & (ds.X <= np.pi))
    assert np.all(ds.subset("test")[2] == -1)
    assert set(np.unique(ds.subset("train")[2])) == {0, 1} and set(np.unique(ds.subset("val")[2])) == {0, 1}
